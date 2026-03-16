//
// Created by Ben Meyers on 2/13/26.
//


#include "ImageComponent.h"
#include "SDL3/SDL.h"
#include "Actor.h"
#include "Game.h"

void ImageComponent::Draw()
{
    if (mTexture == nullptr)
    {
        return;
    }
    auto rect = SDL_FRect();
    SDL_GetTextureSize(mTexture, &rect.w, &rect.h);
    const float scale = GetTransform().GetScale();
    rect.w *= scale;
    rect.h *= scale;

    // if no custom pivot, then the pivot is just the center of the rect
    const SDL_FPoint pivot = mHasCustomPivot ? mPivot : SDL_FPoint{rect.w / 2.0f, rect.h / 2.0f};

    // now set rect's top left origin based on pivot (because transform holds pivot point in world space)
    rect.x = GetTransform().GetPosition().x - pivot.x;
    rect.y = GetTransform().GetPosition().y - pivot.y;

    SDL_RenderTextureRotated(mRenderer, mTexture, nullptr, &rect, GetTransform().GetRotation(), &pivot,
                             SDL_FLIP_NONE);
}

void ImageComponent::SetTexture(SDL_Texture* texture)
{
    mTexture = texture;
}

Transform& ImageComponent::GetTransform()
{
    return mUseLocalTransform ? mLocalTransform : GetOwner()->GetTransform();
}

ImageComponent::ImageComponent( Actor* owner) : Component(owner)
{
    mRenderer = gGame.GetRenderer();
    gGame.AddRenderable(this);
}

ImageComponent::~ImageComponent()
{
    gGame.RemoveRenderable(this);
}

void ImageComponent::HandleRender()
{
    Component::HandleRender();
    Draw();
}