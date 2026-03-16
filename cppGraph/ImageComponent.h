//
// Created by Ben Meyers on 2/13/26.
//

#ifndef RELATIVITY_IMAGECOMPONENT_H
#define RELATIVITY_IMAGECOMPONENT_H
#include "Component.h"
#include <SDL3/SDL.h>
#include "Transform.h"

class ImageComponent : public Component {
    SDL_Texture* mTexture = nullptr;
    SDL_Renderer* mRenderer = nullptr;
    Transform mLocalTransform;
    bool mUseLocalTransform = false;
    bool mHasCustomPivot = false;
    // pivot is object space from top left origin
    SDL_FPoint mPivot{};
public:
    void Draw();
    void SetTexture(SDL_Texture* texture);
    void SetUseLocalTransform(bool use) { mUseLocalTransform = use; }
    void SetPivot(SDL_FPoint pivot) { mHasCustomPivot = true; mPivot = pivot; }
    void ClearPivot() { mHasCustomPivot = false; }
    [[nodiscard]] Transform& GetTransform();
protected:
    explicit ImageComponent(class Actor* owner);
    ~ImageComponent() override;
    void HandleRender() override;
    friend class Actor;
};

#endif