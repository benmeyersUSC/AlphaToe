//
// Created by Ben Meyers on 2/11/26.
//

#include "Actor.h"

#include "Component.h"
#include "Game.h"

Actor::~Actor() {
    for (auto& comp : mComponents)
    {
        delete comp;
    }
    mComponents.clear();
}


void Actor::Update(float deltaTime)
{
    for (auto& comp : mComponents)
    {
        comp->Update(deltaTime);
    }
    HandleUpdate(deltaTime);
}

void Actor::Render() {
    // actor populates its draw components with shapes
    // actual component rendering is handled by Game's sorted render pass
    HandleRender();
}
void Actor::HandleRender() {
}

void Actor::Input(const bool keys[], SDL_MouseButtonFlags mouseButtons,
                  const Vector2& posMouse)
{
    for (auto& c : mComponents)
    {
        c->Input(keys, mouseButtons, posMouse);
    }
    HandleInput(keys, mouseButtons, posMouse);
}

void Actor::Destroy()
{
    gGame.AddPendingDestroy(this);
}

void Actor::HandleUpdate(float deltaTime) {}

void Actor::HandleInput(const bool keys[], SDL_MouseButtonFlags mouseButtons,
                             const Vector2& posMouse) {}


