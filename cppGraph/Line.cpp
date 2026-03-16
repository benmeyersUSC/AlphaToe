//
// Created by Ben Meyers on 2/15/26.
//

#include "Line.h"
#include "DrawComponent.h"
#include "Game.h"

Line::Line() : Actor() {
    mDraw = CreateComponent<DrawComponent>();
}

void Line::HandleRender() {
    const bool steep = Math::Abs(mSlope.y) > Math::Abs(mSlope.x);

    // parametric form
    // given by: x = mOrigin.x + ratio * (y - mOrigin.y)

    float x1, y1, x2, y2;
    if (steep) {
        const float ratio = Math::NearlyZero(mSlope.y) ? 0.0f : mSlope.x / mSlope.y;
        y1 = 0.0f;
        // x coordinate of intercept at top of screen
        x1 = mOrigin.x + ratio * (y1 - mOrigin.y);
        y2 = Game::WINDOW_HEIGHT;
        // x coordinate of intercept at bottom of screen
        x2 = mOrigin.x + ratio * (y2 - mOrigin.y);
    } else {
        const float ratio = Math::NearlyZero(mSlope.x) ? 0.0f : mSlope.y / mSlope.x;
        x1 = 0.0f;
        // y coordinate of intercept at left edge of screen
        y1 = mOrigin.y + ratio * (x1 - mOrigin.x);
        x2 = Game::PLOT_WIDTH;
        // y coordinate of intercept at right edge of screen
        y2 = mOrigin.y + ratio * (x2 - mOrigin.x);
    }

    mDraw->AddLine(x1, y1, x2, y2, Game::MAX_COLOR, Game::MAX_COLOR, Game::MAX_COLOR, Game::MAX_COLOR, mThickness);

    Actor::HandleRender();
}
