#include <utility>

#include "flamegpu/visualiser/PanelVis.h"

namespace flamegpu {
namespace visualiser {

PanelVis::PanelVis(std::shared_ptr<PanelConfig> _m, std::shared_ptr<EnvironmentDescription> _environment)
    : env_properties(_environment->getPropertiesMap())  // For some reason this method returns a copy, not a reference
    , m(std::move(_m)) {
    // Rebuild added_properties
    for (const auto &element : m->ui_elements) {
        if (const EnvPropertyElement* p = dynamic_cast<EnvPropertyElement*>(element.get())) {
            added_properties.insert({p->name, p->index});
        }
    }
}
void PanelVis::newSection(const std::string& header_text, bool begin_open) {
    std::unique_ptr<PanelElement> ptr = std::make_unique<HeaderElement>(header_text, begin_open);
    m->ui_elements.push_back(std::move(ptr));
}
void PanelVis::newEndSection() {
    std::unique_ptr<PanelElement> ptr = std::make_unique<EndSectionElement>();
    m->ui_elements.push_back(std::move(ptr));
}
void PanelVis::newStaticLabel(const std::string& label_text) {
    std::unique_ptr<PanelElement> ptr = std::make_unique<LabelElement>(label_text);
    m->ui_elements.push_back(std::move(ptr));
}
void PanelVis::newSeparator() {
    std::unique_ptr<PanelElement> ptr = std::make_unique<SeparatorElement>();
    m->ui_elements.push_back(std::move(ptr));
}

}  // namespace visualiser
}  // namespace flamegpu
