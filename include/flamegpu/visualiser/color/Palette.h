#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_PALETTE_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_PALETTE_H_

#include <iterator>
#include <vector>
#include <array>
#include <cmath>

#include "flamegpu/visualiser/color/Color.h"
#include "flamegpu/visualiser/color/ViridisInterpolation.h"

namespace flamegpu {
namespace visualiser {

class AutoPalette;

/**
 * Abstract class for representing collections of const colors
 * Optionally, an enum can be added to sub-classes to allow them to be accessed via a name
 */
struct Palette {
    friend class AutoPalette;
    /**
     * The possibles types of palette
     * Qualitative: Distinct/unrelated colours
     * Sequential: Colors in a sequential order, usually best treated as ordering low to high
     * Diverging: Colors that are ordered such towards a central midpoint
     */
    enum Category{ Qualitative, Sequential, Diverging };
    typedef size_t size_type;
    /**
     * Basic iterator that provides constant access to a palettes colors.
     */
    class const_iterator {
        typedef std::input_iterator_tag iterator_category;
        typedef Color value_type;
        typedef size_type difference_type;
        typedef const Color* pointer;
        typedef const Color& reference;

        const Palette& palette;
        difference_type pos;
     public:
        const_iterator(const Palette& _palette, difference_type _pos)
            : palette(_palette), pos(_pos) { }
        const_iterator& operator++() { ++pos; return *this; }
        const_iterator operator++(int) { const_iterator retval = *this; ++(*this); return retval; }
        bool operator==(const_iterator other) const {
            return pos == other.pos && palette == other.palette;
        }
        bool operator!=(const_iterator other) const { return !(*this == other); }
        const Color& operator*() const { return palette[pos]; }
    };
    /**
     * Default virtual destructor
     */
    virtual ~Palette() = default;
    /**
     * Access the members of the palette as an array
     * @param i Index of the desired element, i must be less than size()
     */
    const Color& operator[](size_t i) const { return colors()[i]; }
    /**
     * Returns the number of colors in the palette
     */
    size_t size() const { return colors().size(); }
    /**
     * Returns an iterator to the first item in the palette
     */
    const_iterator begin() const {return const_iterator(*this, 0); }
    /**
     * Returns an iterator to after the last item in the palette
     */
    const_iterator end() const { return const_iterator(*this, size()); }
    /**
     * Compares two palettes for equality (whether they contain the same colors)
     */
    bool operator==(const Palette& other) const {
        if (size() != other.size())
            return false;
        for (unsigned int i = 0; i < size(); ++i)
            if ((*this)[i] != other[i])
                return false;
        return true;
    }
    /**
     * Returns whether the palette is confirmed as suitable for colorblind viewers
     */
    virtual bool getColorBlindFriendly() const = 0;
    /**
     * Returns the category of the palette
     * @see Category
     */
    virtual Category getCategory() const = 0;

 protected:
    virtual const std::vector<Color>& colors() const = 0;
};


namespace Stock {
/**
 * A collection of preset palettes
 * Mostly copied from existing libraries, e.g. Colorbrewer, Seaborn
 */
namespace Palettes {
/**
 * Qualitative palette
 * Set1 from Colorbrewer
 */
struct Set1 : Palette {
    Set1() { }
    Category getCategory() const override { return Qualitative; }
    bool getColorBlindFriendly() const override { return false; }
    /**
     * Friendly names that can be passed to operator[]() to retrieve the relevant color
     */
    enum Name {
        RED,
        BLUE,
        GREEN,
        PURPLE,
        ORANGE,
        YELLOW,
        BROWN,
        PINK,
        GREY
    };

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("E41A1C"),
            Color("377EB8"),
            Color("4DAF4A"),
            Color("984EA3"),
            Color("FF7F00"),
            Color("FFFF33"),
            Color("A65628"),
            Color("F781BF"),
            Color("999999"),
        };
        return colors;
    }
};
/**
 * Color blind friendly qualitative palette
 * Set2 from Colorbrewer
 * @note Color names are approximations using https://www.color-blindness.com/color-name-hue/
 */
struct Set2 : Palette {
    Set2() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Qualitative; }
    bool getColorBlindFriendly() const override { return true; }
    /**
     * Friendly names that can be passed to operator[]() to retrieve the relevant color
     */
    enum Name {
        PUERTO_RICO,
        ATOMIC_TANGERINE,
        POLO_BLUE,
        SHOCKING,
        CONIFER,
        SUNGLOW,
        CHAMOIS,
        DARK_GREY,
    };

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("66C2A5"),
            Color("FC8D62"),
            Color("8DA0CB"),
            Color("E78AC3"),
            Color("A6D854"),
            Color("FFD92F"),
            Color("E5C494"),
            Color("B3B3B3"),
        };
        return colors;
    }
};
/**
 * Color blind friendly qualitative palette
 * Dark2 from Colorbrewer
 * @note Color names are approximations using https://www.color-blindness.com/color-name-hue/
 */
struct Dark2 : Palette {
    Dark2() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Qualitative; }
    bool getColorBlindFriendly() const override { return true; }
    /**
     * Friendly names that can be passed to operator[]() to retrieve the relevant color
     */
    enum Name {
        ELF_GREEN,
        TAWNY,
        RICH_BLUE,
        RAZZMATAZZ,
        CHRISTI,
        GAMBOGE,
        GOLDEN_BROWN,
        MORTAR,
    };

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("1D8F64"),
            Color("CE4A08"),
            Color("6159A4"),
            Color("DE0077"),
            Color("569918"),
            Color("DF9C09"),
            Color("946317"),
            Color("535353"),
        };
        return colors;
    }
};
/**
 * Qualitative palette
 * pastel palette from seaborn
 * Blue light filters may cause MACARONI_AND_CHEESE(1) and ROSEBUD(3) to appear similar
 * @note Color names are approximations using https://www.color-blindness.com/color-name-hue/
 */
struct Pastel : Palette {
    Pastel() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Qualitative; }
    bool getColorBlindFriendly() const override { return false; }
    /**
     * Friendly names that can be passed to operator[]() to retrieve the relevant color
     */
    enum Name {
        PALE_CORNFLOWER_BLUE,
        MACARONI_AND_CHEESE,
        GRANNY_SMITH_APPLE,
        ROSEBUD,
        MAUVE,
        PANCHO,
        LAVENDER_ROSE,
        VERY_LIGHT_GREY,
        CANARY,
        PALE_TURQUOISE,
    };

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("A1C9F4"),
            Color("FFB482"),
            Color("8DE5A1"),
            Color("FF9F9B"),
            Color("D0BBFF"),
            Color("DEBB9B"),
            Color("FAB0E4"),
            Color("CFCFCF"),
            Color("FFFEA3"),
            Color("B9F2F0"),
        };
        return colors;
    }
};
/**
 * Color blind friendly Sequential palette
 * YlOrRd palette from Colorbrewer
 */
struct YlOrRd : Palette {
    YlOrRd() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Sequential; }
    bool getColorBlindFriendly() const override { return true; }

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("FFFFCC"),
            Color("FFEDA0"),
            Color("FED976"),
            Color("FEB24C"),
            Color("FD8D3C"),
            Color("FC4E2A"),
            Color("E31A1C"),
            Color("BD0026"),
            Color("800026"),
        };
        return colors;
    }
};
/**
 * Color blind friendly Sequential palette
 * YlGn palette from Colorbrewer
 */
struct YlGn : Palette {
    YlGn() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Sequential; }
    bool getColorBlindFriendly() const override { return true; }

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("FFFFE5"),
            Color("EDF8B1"),
            Color("D9F0A3"),
            Color("ADDD8E"),
            Color("78C679"),
            Color("41AB5D"),
            Color("238443"),
            Color("006837"),
            Color("004529"),
        };
        return colors;
    }
};
/**
 * Color blind friendly Sequential palette
 * Greys palette from Colorbrewer
 */
struct Greys : Palette {
    Greys() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Sequential; }
    bool getColorBlindFriendly() const override { return true; }

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("FFFFFF"),
            Color("F0F0F0"),
            Color("D9D9D9"),
            Color("BDBDBD"),
            Color("969696"),
            Color("737373"),
            Color("525252"),
            Color("252525"),
            Color("000000"),
        };
        return colors;
    }
};
/**
 * Color blind friendly Diverging palette
 * RdYlBu palette from Colorbrewer
 */
struct RdYlBu : Palette {
    RdYlBu() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Diverging; }
    bool getColorBlindFriendly() const override { return true; }

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("A50026"),
            Color("D73027"),
            Color("ED8160"),
            Color("FDAE61"),
            Color("FEE090"),
            Color("FFFFBF"),
            Color("E0F3F8"),
            Color("ABD9E9"),
            Color("74ADD1"),
            Color("4575B4"),
            Color("313695"),
        };
        return colors;
    }
};
/**
 * Color blind friendly Diverging palette
 * PiYG palette from Colorbrewer
 */
struct PiYG : Palette {
    PiYG() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Diverging; }
    bool getColorBlindFriendly() const override { return true; }

 protected:
    const std::vector<Color>& colors() const override {
        static auto colors = std::vector<Color>{
            Color("8E0152"),
            Color("C51B7D"),
            Color("DE77AE"),
            Color("F1B6DA"),
            Color("FDE0EF"),
            Color("F7F7F7"),
            Color("E6F5D0"),
            Color("B8E186"),
            Color("7FBC41"),
            Color("4D9221"),
            Color("276419"),
        };
        return colors;
    }
};
/**
 * Color blind friendly dynamic sequential palette
 * Viridis from BIDS/MatPlotLib: https://github.com/BIDS/colormap
 */
struct Viridis : Palette {
    Viridis() { }  // Empty default constructor for warning suppression
    Category getCategory() const override { return Sequential; }
    bool getColorBlindFriendly() const override { return true; }
    /**
     * Construct the Palette by specifying how many color values are required
     */
    explicit Viridis(const unsigned int& size)
        : data(initColors(size)) { }

 protected:
    const std::vector<Color>& colors() const override { return data; }

 private:
    const std::vector<Color> data;
    std::vector<Color> initColors(const unsigned int& size) {
        const std::array<const Color, 256>  &raw_colors = ViridisInterpolation::rawColors();
        std::vector<Color> rtn;
        float x = 0.0f;
        for (unsigned int i = 0; i < size; ++i) {
            const float a = std::floor(x);
            const float t = x - a;
            const Color& c0 = raw_colors[static_cast<size_t>(a)];
            const Color& c1 = raw_colors[static_cast<size_t>(std::ceil(x))];
            rtn.push_back(c0 * (1.0f - t) + c1 * t);
            // Move X to next value
            x += raw_colors.size() / static_cast<float>(size - 1);
            x = i == size - 2 ? raw_colors.size() - 1 : x;
        }
        return rtn;
    }
};
/**
 * Qualitative Instances
 */
static const Set1 SET1;
static const Set2 SET2;
static const Dark2 DARK2;
static const Pastel PASTEL;
/**
 * Sequential Instances
 */
static const YlOrRd YLORRD;
static const YlGn YLGN;
static const Greys GREYS;
/**
 * Diverging Instances
 */
static const RdYlBu RDYLBU;
static const PiYG PIYG;
}  // namespace Palettes
}  // namespace Stock

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_PALETTE_H_
