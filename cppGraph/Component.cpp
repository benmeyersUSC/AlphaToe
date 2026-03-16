//
// Created by Ben Meyers on 2/13/26.
//

#include "Component.h"
#include "Game.h"

Component::Component(Actor* owner)
{
    mOwner = owner;
}
Component::~Component()
{
}
void Component::HandleUpdate(float deltaTime)
{
}

void Component::Render() {
    HandleRender();
}

void Component::HandleRender() {
}

void Component::HandleInput(const bool keys[], SDL_MouseButtonFlags mouseButtons,
                            const Math::Vector2& relativeMouse)
{
}
Actor* Component::GetOwner() const
{
    return mOwner;
}
void Component::Update(float deltaTime)
{
    HandleUpdate(deltaTime);
}
void Component::Input(const bool keys[], SDL_MouseButtonFlags mouseButtons,
                      const Math::Vector2& relativeMouse)
{
    HandleInput(keys, mouseButtons, relativeMouse);
}
void Component::SetDrawOrder(int order)
{
    gGame.RemoveRenderable(this);
    mDrawOrder = order;
    gGame.AddRenderable(this);
}