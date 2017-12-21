package org.cvut.havluja1.eggdetector;

class RectF {
    public float left;
    public float top;
    public float right;
    public float bottom;

    RectF() {}

    RectF(float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    RectF(RectF r) {
        if (r == null) {
            left = top = right = bottom = 0.0f;
        } else {
            left = r.left;
            top = r.top;
            right = r.right;
            bottom = r.bottom;
        }
    }

    public String toString() {
        return "RectF(" + left + ", " + top + ", "
                + right + ", " + bottom + ")";
    }

    final float width() {
        return right - left;
    }

    final float height() {
        return bottom - top;
    }

    final float centerX() {
        return (left + right) * 0.5f;
    }

    final float centerY() {
        return (top + bottom) * 0.5f;
    }
}
