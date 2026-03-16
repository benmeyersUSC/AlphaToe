//
// Created by Ben Meyers on 2/11/26.
//

#ifndef RELATIVITY_GAME_H
#define RELATIVITY_GAME_H

#pragma once

#include "SDL3/SDL.h"
#include "Math.h"
#include <functional>
#include <vector>
#include <unordered_map>
#include "Actor.h"
class SpacetimeReference;
class SpacetimePlayer;

class SpacetimeDisplay;
class Component;
class DrawComponent;
class ReferenceActor;
using Math::Vector2;
class Player;

class Game
{
public:

	static constexpr auto BUDDHA_FILE = "Assets/Buddha.png";
	static constexpr auto ARROW_FILE = "Assets/Arrow.png";

	static constexpr float DURATION_SECONDS = 10.0f;
	static constexpr float MAX_VELO = 1000.0f;

	static constexpr unsigned ESTIMATED_FPS = 60;

	static constexpr float PLOT_POINT_SIZE = 5.0f;
	static constexpr float HALF_PLOT_POINT_SIZE = PLOT_POINT_SIZE/2.0f;
	// window constants
	static constexpr float PLOT_WIDTH = 1107.0f;
	static constexpr float SIDEBAR_WIDTH = 363.0f;
	static constexpr float WINDOW_WIDTH = PLOT_WIDTH + SIDEBAR_WIDTH;
	static constexpr float WINDOW_HEIGHT = 855.0f;
	static constexpr float HALF_WIDTH = PLOT_WIDTH/2.0f;
	static constexpr float HALF_HEIGHT = WINDOW_HEIGHT/2.0f;

	// max deltaTime for update
	static constexpr float MAX_DELTA_TIME = 0.033f;

	// constant for full color
	static constexpr Uint8 MAX_COLOR = 255;

	// constant for half size of character pixel count (according to SDL)
	static constexpr float CHAR_PIXELS = 8.0f;
	static constexpr float HALF_CHAR_PIXELS = 4.0f;

	// constant for 1000 ms in one s
	static constexpr float MS_PER_SEC = 1000.0f;



	Game();

	// Initialize the game
	// Returns true if successful
	bool Initialize();

	// Runs an interation of the game loop
	// Returns true if the game loop should continue
	bool RunIteration();

	// Called when the game gets shutdown
	void Shutdown();

	// Called when the game receives an event from SDL
	void HandleEvent(const SDL_Event* event);

	template<typename A>
	A* CreateActor() {
		A* a = new A();
		mPendingCreate.push_back(a);
		return a;
	}

	void Restart();
	void RequestRestart();
	static void LeadingEdge(bool cur, bool& last, const std::function<void()>& fn, bool condition = true);
	void AddPendingDestroy(class Actor* actor);

	void AddRenderable(Component* comp);
	void RemoveRenderable(Component* comp);

	[[nodiscard]] const Vector2& GetMousePos()const{return mMousePos;}

	SDL_Texture* GetTexture(std::string_view filename);
	SDL_Renderer* GetRenderer(){return mSdlRenderer;}
	[[nodiscard]] float GetDT()const{return mDT;}
	SpacetimeDisplay* GetSpacetimeDisplay(){return mSpacetimeDisplay;}

	[[nodiscard]] float ElapsedPct()const;
private:
	// window and renderer
	SDL_Window* mSdlWindow;
	SDL_Renderer* mSdlRenderer;
	std::unordered_map<std::string, SDL_Texture*> mTextures;

	// keep the game going
	bool mContinueRunning;
	bool mGameDone = false;
	bool mRestartPending = false;
	bool mLastR = false;

	std::function<void()> mOnRestart = [this]{ mRestartPending = true; };

	// aggregation factor, computed after simulation ends
	unsigned mFrameAgg = 1;

	// actors vector and individual member variables
	std::vector<Actor*> mActors;
	std::vector<Actor*> mPendingCreate;
	std::vector<Actor*> mPendingDestroy;
	std::vector<Component*> mRenderables;
	Player* mPlayer;

	void DestroyActor(Actor* actor);


	// prev time for delta calcs
	Uint64 mPreviousTime;

	void ProcessInput();
	void UpdateGame();
	void GenerateOutput();

	void LoadData();
	void UnloadData();

	void EndGame();

	static void TransformPoints(const std::vector<float>& src, std::vector<float>& dest, const std::function<float(float)>& transformFunc, unsigned frameAgg, bool notTime=true) ;
	// these should return [-WIDTH/2, WIDTH/2]
	std::function<float(float)> mTransformPosition = [](const float position) {
		return position - PLOT_WIDTH/2.0f;
	};

	// lowkey this should be scaled to something else
	std::function<float(float)> mTransformVelocity = [](const float velo) {
		// percentage of max velo, scaled to WIDTH/2
		return velo * PLOT_WIDTH / (MAX_VELO * 2.0f);
	};
	std::function<float(float)> mTransformTimeVelocity = [](float f){
		// dτ/dt ratio [0, 1]
		return f / MAX_VELO;
	};

	float mDT = 0.0f;
	Vector2 mMousePos;
	size_t mTimestepIndex = 0;
	SpacetimeDisplay* mSpacetimeDisplay = nullptr;
	SpacetimeReference* mSpacetimeBuddha = nullptr;
	SpacetimePlayer* mSpacetimePlayer = nullptr;
};

extern Game gGame;

#endif //RELATIVITY_GAME_H