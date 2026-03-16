//
// Created by Ben Meyers on 2/11/26.
//

#include "Transform.h"

SDL_FRect Transform::GetRect() const
{
    return SDL_FRect(
        // compensate for middle-storage of coordinates
        mPosition.x - mSize.x / 2.0f, mPosition.y - mSize.y / 2.0f, mSize.x, mSize.y);
}