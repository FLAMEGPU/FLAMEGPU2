#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_AUTOPALETTE_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_AUTOPALETTE_H_

#include <vector>

#include "flamegpu/visualiser/color/Color.h"
#include "flamegpu/visualiser/color/Palette.h"

namespace flamegpu {
namespace visualiser {

/**
 * Automatically iterates the colors of a Palette on next()
 */
class AutoPalette {
    const std::vector<Color> palette;
    size_t next_index;
 public:
    /**
     * Construct a new AutoPalette, with the colors from palette
     * @param _palette Palette of colors to iterate
     * @note A copy of the palette's colors are taken
     */
    explicit AutoPalette(const Palette& _palette)
        : palette(_palette.colors())
        , next_index(0) { }
    /**
     * Returns the next color
     * If palette.size() is exceed, returns to the first color
     */
    const Color &next() {
        if (next_index >= palette.size())
            next_index = 0;
        return palette[next_index++];
    }
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_AUTOPALETTE_H_
