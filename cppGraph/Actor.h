//
// Created by Ben Meyers on 2/11/26.
//

#ifndef RELATIVITY_ACTOR_H
#define RELATIVITY_ACTOR_H
#include <vector>

#include "Transform.h"
#include "SDL3/SDL_mouse.h"
class Component;
class Actor {
public:
    virtual ~Actor();

    Transform& GetTransform() { return mTransform; }
    [[nodiscard]] const Transform& GetTransform() const { return mTransform; }

    template <typename T>
    T* CreateComponent()
    {
        // 'this' is the owner (an Actor) for the Component (required in Component constructor)
        T* component = new T(this);

        mComponents.emplace_back(component);

        return component;
    }

    // Returns component of type T, or nullptr if it doesn't exist
    template <typename T>
    T* GetComponent() const
    {
        // Loop over all components
        for (auto c : mComponents)
        {
            // dynamic_cast will return nullptr if c is not type T*
            T* t = dynamic_cast<T*>(c);
            if (t)
            {
                return t;
            }
        }

        return nullptr;
    }

    void Update(float deltaTime);
    void Render();
    void Input(const bool keys[], SDL_MouseButtonFlags mouseButtons, const Vector2& posMouse);
    void Destroy();

    explicit Actor() = default;
protected:
    virtual void HandleRender();
    virtual void HandleUpdate(float deltaTime);
    virtual void HandleInput(const bool keys[], SDL_MouseButtonFlags mouseButtons,
                             const Vector2& posMouse);

private:
    Transform mTransform;
    std::vector<Component*> mComponents;
};


#endif //RELATIVITY_ACTOR_H