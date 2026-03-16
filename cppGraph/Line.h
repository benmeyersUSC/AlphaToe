//
// Created by Ben Meyers on 2/15/26.
//

#ifndef RELATIVITY_LINE_H
#define RELATIVITY_LINE_H

#include "Actor.h"
#include "Game.h"
class DrawComponent;

// this class defines Lines in the mathematical sense; not segments.
// they span the entire screen and intersect at some point with some slope
class Line : public  Actor{
    // this is SDL slope (pos y is down)
    Vector2 mSlope;
    // point through which to pass
    Vector2 mOrigin;
    int mThickness = 1;
    DrawComponent* mDraw = nullptr;
public:
    static constexpr Vector2 X_AXIS_SLOPE{1.0f, 0.0f};
    static constexpr Vector2 X_AXIS_ORIGIN{0.0f, Game::HALF_HEIGHT};
    static constexpr Vector2 Y_AXIS_SLOPE{0.0f, -1.0f};
    static constexpr Vector2 Y_AXIS_ORIGIN{Game::HALF_WIDTH, 0.0f};

    Line();
    void SetSlope(const Vector2& slope) {mSlope = slope;}
    void SetOrigin (const Vector2& og) {mOrigin = og;}
    void SetThickness(const int t){mThickness = t;}
protected:
    void HandleRender() override;
};


#endif //RELATIVITY_LINE_H