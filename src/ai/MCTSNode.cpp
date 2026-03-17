#include "MCTSNode.h"

MCTSNode::~MCTSNode() {
    for (auto* child : mChildren)
    {
        delete child;
    }
}
