#include "Game.h"

#include <iostream>
#include "Actor.h"
#include "Player.h"
#include <functional>
#include "SDL3_image/SDL_image.h"
#include "SpacetimePlayer.h"
#include "SpacetimeComponent.h"
#include "ImageComponent.h"

#include "SpacetimeReference.h"
#include "Component.h"
#include "DrawComponent.h"
#include "Line.h"
#include "SpacetimeDisplay.h"
Game gGame;

Game::Game()
{
	// comments in class declaration explain these members

	mSdlWindow = nullptr;
	mSdlRenderer = nullptr;

	mContinueRunning = true;

	// actors start null
	mPlayer = nullptr;

	mPreviousTime = 0;
}

bool Game::Initialize()
{
	SDL_SetHint("SDL_MAIN_CALLBACK_RATE", "60");
	bool sdlInit = SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
	// fail if SDL init doesn't work
	if (!sdlInit)
	{
		return false;
	}
	// use window constants
	mSdlWindow = SDL_CreateWindow("SPECIAL RELATIVITY", PLOT_WIDTH, WINDOW_HEIGHT, 0);
	if (mSdlWindow == nullptr)
	{
		return false;
	}

	mSdlRenderer = SDL_CreateRenderer(mSdlWindow, nullptr);
	if (mSdlRenderer == nullptr)
	{
		return false;
	}

	// init actors!
	LoadData();
	return true;
}

bool Game::RunIteration()
{
	// game loop!

	// get input
	ProcessInput();
	// update game objects
	UpdateGame();
	// render
	GenerateOutput();
	return mContinueRunning;
}

void Game::Shutdown()
{
	// call proper unloaders/destroyers
	UnloadData();
	SDL_DestroyRenderer(mSdlRenderer);
	SDL_DestroyWindow(mSdlWindow);
	SDL_Quit();
}

void Game::HandleEvent(const SDL_Event* event)
{
	// x button
	if (event->type == SDL_EVENT_QUIT)
	{
		mContinueRunning = false;
	}
}

void Game::DestroyActor(Actor *actor) {

		std::erase(mActors, actor);
		delete actor;

}

void Game::ProcessInput()
{
	const bool* keyboardState = SDL_GetKeyboardState(nullptr);
	// esc key can also end game
	if (keyboardState[SDL_SCANCODE_ESCAPE])
	{
		mContinueRunning = false;
	}

	LeadingEdge(keyboardState[SDL_SCANCODE_R], mLastR, mOnRestart, mGameDone);

	// mouse
	SDL_MouseButtonFlags mouseButtons = SDL_GetMouseState(&mMousePos.x, &mMousePos.y);

	for (auto a : mActors) {
		a->Input(keyboardState, mouseButtons, mMousePos);
	}
	if (mSpacetimeDisplay){
		mTimestepIndex = mSpacetimeDisplay->GetTimestepIndex();
	}
}

void Game::TransformPoints(const std::vector<float>& src, std::vector<float>& dest, const std::function<float(float)>& transformFunc, unsigned frameAgg, bool notTime) {
	// go agg group by agg group
	for (size_t i = 0; i + frameAgg <= src.size(); i += frameAgg) {
		// average the group
		float sum = 0.0f;
		for (size_t x = 0; x < frameAgg; ++x) {
			sum += src[i + x];
		}
		// and apply the transformation function and center around mid if not time
		// now these are transformed for rendering
		dest.push_back(transformFunc(sum/static_cast<float>(frameAgg)) + static_cast<float>(notTime) * HALF_WIDTH);
	}
}

void Game::AddRenderable(Component* comp) {
	const auto it = std::ranges::upper_bound(mRenderables, comp,
	                                   [](const Component* a, const Component* b) {
		                                   return a->GetDrawOrder() < b->GetDrawOrder();
	                                   });
	mRenderables.insert(it, comp);
}

void Game::RemoveRenderable(Component* comp) {
	std::erase(mRenderables, comp);
}

void Game::AddPendingDestroy(class Actor *actor) {
	if (std::ranges::find(mPendingDestroy, actor) == mPendingDestroy.end())
	{
		mPendingDestroy.emplace_back(actor);
	}
}


SDL_Texture * Game::GetTexture(std::string_view filename) {

	auto it = mTextures.find(filename.data());
	if (it != mTextures.end())
	{
		return it->second;
	}
	std::string fullPath = std::string(SDL_GetBasePath()) + std::string(filename);
	SDL_Surface* imgSurface = IMG_Load(fullPath.c_str());
	if (imgSurface == nullptr)
	{
		SDL_Log("Image file: %s failed to load!", filename.data());
		return nullptr;
	}
	SDL_Texture* imgTexture = SDL_CreateTextureFromSurface(mSdlRenderer, imgSurface);
	SDL_SetTextureBlendMode(imgTexture, SDL_BLENDMODE_BLEND);
	SDL_DestroySurface(imgSurface);

	mTextures[filename.data()] = imgTexture;
	return imgTexture;
}

float Game::ElapsedPct() const {
	return mDT / DURATION_SECONDS;
}


void Game::EndGame() {
	auto ppos = std::move(mPlayer->GetSpacetime()->GetSpatialPositions());
	auto pv = std::move(mPlayer->GetSpacetime()->GetSpatialVelocities());
	auto ptv = std::move(mPlayer->GetSpacetime()->GetProperTimes());
	unsigned numSamples = ppos.size();
	auto maxPlotPoints = static_cast<unsigned>(WINDOW_HEIGHT / PLOT_POINT_SIZE);
	// how many samples will we average to meet the ideal/max plot points
	auto frameAgg = std::max(1u, numSamples/ maxPlotPoints);
	// destroy actors and ditch paddle
	UnloadData();
	mPlayer = nullptr;


	// expand window to fit sidebar and re-center on screen
	SDL_SetWindowSize(mSdlWindow, WINDOW_WIDTH, WINDOW_HEIGHT);
	SDL_SetWindowPosition(mSdlWindow, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

	mSpacetimeDisplay = CreateActor<SpacetimeDisplay>();

	// transform raw samples...extract from actor(s) before we destroy them, pass along to spacetime plot
	// - scaled by width
	// - mapped via aggregation to vertical plotting space!
	TransformPoints(ppos, mSpacetimeDisplay->GetPositions(), mTransformPosition, frameAgg);
	TransformPoints(pv, mSpacetimeDisplay->GetVelocities(), mTransformVelocity, frameAgg);
	TransformPoints(ptv, mSpacetimeDisplay->GetProperTimes(), mTransformTimeVelocity,frameAgg, false);

	mSpacetimeDisplay->Setup();
	auto sidebar = CreateActor<Line>();
	sidebar->SetOrigin({PLOT_WIDTH, HALF_HEIGHT});
	sidebar->SetSlope(Line::Y_AXIS_SLOPE);

	mSpacetimeBuddha = CreateActor<SpacetimeReference>();
	mSpacetimePlayer = CreateActor<SpacetimePlayer>();
}

void Game::UpdateGame()
{
	if (mRestartPending)
		return Restart();

	// calculate deltatime
	Uint64 currTimeMs = SDL_GetTicks();
	Uint64 uIntDiff = currTimeMs - mPreviousTime;
	mPreviousTime = currTimeMs;
	float deltaTime = static_cast<float>(uIntDiff) / MS_PER_SEC;
	deltaTime = Math::Min(MAX_DELTA_TIME, deltaTime);
	mDT += deltaTime;

	// 'leading edge' of elapsed eclipsing set duration
	if (mDT >= DURATION_SECONDS && !mGameDone) {
		mGameDone = true;
		return EndGame();
	}

	for (auto actor : mPendingCreate)
		mActors.emplace_back(actor);

	mPendingCreate.clear();

	for (auto actor : mActors)
		actor->Update(deltaTime);

	for (auto actor : mPendingDestroy)
		DestroyActor(actor);

	mPendingDestroy.clear();
}

void Game::GenerateOutput()
{
	SDL_SetRenderDrawColor(mSdlRenderer, 0, 0, 0, MAX_COLOR);
	SDL_RenderClear(mSdlRenderer);

	// actors call their own HandleRender, creating shapes
	for (auto& a : mActors) {
		a->Render();
	}

	// actual rendering components then draw shapes to SDL in order
	for (auto& comp : mRenderables) {
		comp->Render();
	}

	SDL_RenderPresent(mSdlRenderer);
}

void Game::LoadData()
{
	auto* xAxis = CreateActor<Line>();
	xAxis->SetOrigin(Line::X_AXIS_ORIGIN);
	xAxis->SetSlope(Line::X_AXIS_SLOPE);

	auto refActor = CreateActor<Actor>();
	refActor->GetTransform().SetScale(0.1f);
	refActor->GetTransform().SetPosition({Game::HALF_WIDTH, Game::HALF_HEIGHT});
	auto refImage = refActor->CreateComponent<ImageComponent>();
	refImage->SetTexture(gGame.GetTexture(Game::BUDDHA_FILE));

	mPlayer = CreateActor<Player>();
	mPlayer->GetTransform().SetPosition(Vector2(HALF_WIDTH, HALF_HEIGHT ));
	mPlayer->GetSpacetime()->Setup(DURATION_SECONDS * ESTIMATED_FPS);
}

void Game::UnloadData()
{
	for (auto& actor : mActors)
		delete actor;
	mActors.clear();

	for (auto& actor : mPendingCreate)
		delete actor;
	mPendingCreate.clear();
}

void Game::RequestRestart()
{
	mRestartPending = true;
}

void Game::LeadingEdge(bool cur, bool& last, const std::function<void()>& fn, bool condition)
{
	if (!last && cur && condition)
		fn();
	last = cur;
}

void Game::Restart()
{
	mRestartPending = false;
	UnloadData();

	mPlayer = nullptr;
	mSpacetimeDisplay = nullptr;
	mSpacetimeBuddha = nullptr;
	mSpacetimePlayer = nullptr;
	mDT = 0.0f;
	mGameDone = false;
	mTimestepIndex = 0;
	mPendingDestroy.clear();
	mPreviousTime = SDL_GetTicks();

	// shrink window back to game width
	SDL_SetWindowSize(mSdlWindow, static_cast<int>(PLOT_WIDTH), static_cast<int>(WINDOW_HEIGHT));
	SDL_SetWindowPosition(mSdlWindow, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

	LoadData();
}

