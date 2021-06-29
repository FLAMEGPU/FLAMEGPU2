#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLOR_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLOR_H_

#include <cstring>
#include <array>

#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace visualiser {

class StaticColor;
/**
 * Store for a floating point rgba color
 * Each component should be in the inclusive range [0, 1] in-order for the color to be considered valid
 */
struct Color {
    /**
     * Color components: red, green, blue, alpha
     */
    float r, g, b, a;
    /**
     * Default constructor, initialises to white
     */
    Color()
        : r(1.0f), g(1.0f), b(1.0f), a(1.0f) { }
    /**
     * Construct a color with red, green, blue (and alpha) components
     * Each component should be in the range [0, 1]
     */
    Color(float _r, float _g, float _b, float _a = 1.0f)
        : r(_r), g(_g), b(_b), a(_a) { }
    Color(double _r, double _g, double _b, double _a = 1.0)
        : r(static_cast<float>(_r)), g(static_cast<float>(_g)), b(static_cast<float>(_b)), a(static_cast<float>(_a)) { }
    /**
     * Construct a color with red, green, blue (and alpha) components
     * Each component should be in the range [0, 255]
     */
    Color(int _r, int _g, int _b, int _a = 255)
        : r(_r / 255.0f), g(_g / 255.0f), b(_b / 255.0f), a(_a / 255.0f) { }
    /**
     * Construct a color with red, green and blue components
     * Each component should be in the range [0, 1]
     */
    explicit Color(const std::array<float, 3>& rgb)
        : r(rgb[0]), g(rgb[1]), b(rgb[2]), a(1.0f) { }
    /**
     * Construct a color with red, green, blue and alpha components
     * Each component should be in the range [0, 1]
     */
    explicit Color(const std::array<float, 4>& rgba)
        : r(rgba[0]), g(rgba[1]), b(rgba[2]), a(rgba[3]) { }
    /**
     * Construct a color with red, green and blue components
     * Each component should be in the range [0, 255]
     */
    explicit Color(const std::array<int, 3> &rgb)
        : r(rgb[0] / 255.0f), g(rgb[1] / 255.0f), b(rgb[2] / 255.0f), a(1.0f) { }
    /**
     * Construct a color with red, green, blue and alpha components
     * Each component should be in the range [0, 255]
     */
    explicit Color(const std::array<int, 4>& rgba)
        : r(rgba[0] / 255.0f), g(rgba[1] / 255.0f), b(rgba[2] / 255.0f), a(rgba[3] / 255.0f) { }
    /**
     * Construct a color from a hexcode
     * Supported formats:
     *     #%abcdef
     *     #%abc
     *     abcdef
     *     abc
     * @throws exception::InvalidArgument If parsing fails
     */
    explicit Color(const char* hex) { *this = hex; }
    /**
     * Construct a color from a hexcode
     * Supported formats:
     *     #%abcdef
     *     #%abc
     *     abcdef
     *     abc
     * @throws exception::InvalidArgument If parsing fails
     */
    Color& operator=(const char *hex) {
        a = 1.0f;
        // Would be nice to get rid of sscanf, so that it could be constexpr
        if (hex[0] == '#') ++hex;
        const size_t hex_len = strlen(hex);
        if (hex_len == 8) {
            int _r, _g, _b, _a;
            const int ct = sscanf(hex, "%02x%02x%02x%02x", &_r, &_g, &_b, &_a);
            if (ct == 4) {
                r = _r / 255.0f; g = _g / 255.0f; b = _b / 255.0f; a = _a / 255.0f;
                return *this;
            }
        } else if (hex_len == 6) {
            int _r, _g, _b;
            const int ct = sscanf(hex, "%02x%02x%02x", &_r, &_g, &_b);
            if (ct == 3) {
                r = _r / 255.0f; g = _g / 255.0f; b = _b / 255.0f;
                return *this;
            }
        } else if (hex_len == 4) {
            int _r, _g, _b, _a;
            const int ct = sscanf(hex, "%01x%01x%01x%01x", &_r, &_g, &_b, &_a);
            if (ct == 4) {
                r = 17.0f * _r / 255.0f; g = 17.0f * _g / 255.0f; b = 17.0f * _b / 255.0f; a = 17.0f * _a / 255.0f;
                return *this;
            }
        } else if (hex_len == 3) {
            int _r, _g, _b;
            const int ct = sscanf(hex, "%01x%01x%01x", &_r, &_g, &_b);
            if (ct == 3) {
                r = 17.0f * _r / 255.0f; g = 17.0f * _g / 255.0f; b = 17.0f * _b / 255.0f;
                return *this;
            }
        }
        THROW exception::InvalidArgument("Unable to parse hex string '%s', must be a string of either 3, 4, 6 or 8 hexidecimal characters, "
            "in Color::Color().\n",
            hex);
    }
    /**
     * Access the color component at index
     */
    float& operator[](unsigned int index) {
        if (index >= 4) {
            THROW exception::InvalidArgument("index '%u' is not in the inclusive range [0, 3], "
                "in Color::operator[]().\n",
                index);
        }
        return (&r)[index];
    }
    /**
     * Access the color component at index
     */
    float operator[](unsigned int index) const {
        if (index >= 4) {
            THROW exception::InvalidArgument("index '%u' is not in the inclusive range [0, 3], "
                "in Color::operator[]().\n",
                index);
        }
        return (&r)[index];
    }
    /**
     * Return true if all color components are in the inclusive range [0.0, 1.0]
     */
    bool validate() const {
        for (unsigned int i = 0; i < 4; ++i)
            if ((&r)[i] < 0.0f || ((&r)[i] > 1.0f))
                return false;
        return true;
    }
    /**
     * Equality operator
     * @note It performs equality comparison on floats, this isn't too useful as very similar floats can be deemed different
     */
    bool operator==(const Color &other) const {
        if (r != other.r ||
            g != other.g ||
            b != other.b ||
            a != other.a)
                return false;
        return true;
    }
    /**
     * Inequality operator
     * @note It performs equality comparison on floats, this isn't too useful as very similar floats can be deemed different
     */
    bool operator!=(const Color& other) const {
        return !(*this == other);
    }
    /**
     * Add the components of 2 colors
     */
    Color operator+(const Color& other) const noexcept {
        return Color{ other.r + r, other.g + g, other.b + b, other.a + a };
    }
    /**
     * Multiple the components of a color by i
     * @todo Rework this to the other format, so color doesnt need to be the first item
     */
    Color operator*(const float &i) const noexcept {
        return Color{ i * r, i * g, i * b, i*a };
    }
    /**
     * Implicit conversion function, allowing Color to be implicitly converted to StaticColor
     * Defined in StaticColor.cpp
     */
    operator StaticColor() const;
};

namespace Stock {
/**
 * A collection of preset colors
 */
namespace Colors {
static const Color BLACK = Color{0.0f, 0.0f, 0.0f};
static const Color WHITE = Color{1.0f, 1.0f, 1.0f};
static const Color RED =   Color{1.0f, 0.0f, 0.0f};
static const Color GREEN = Color{0.0f, 1.0f, 0.0f};
static const Color BLUE =  Color{0.0f, 0.0f, 1.0f};
}  // namespace Colors
}  // namespace Stock

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLOR_H_
