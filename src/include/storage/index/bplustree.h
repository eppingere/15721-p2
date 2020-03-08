#pragma once

#include <common/shared_latch.h>
#include <execution/util/execution_common.h>
#include <storage/storage_defs.h>
#include <functional>
#include <utility>
#include <vector>

#include "common/macros.h"

namespace terrier::storage::index {

/*
 * class BPlusTree - Latch Crabbing BPlusTree index implementation
 *
 * Template Arguments:
 *
 * template <typename KeyType,
 *           typename ValueType,
 *           typename KeyComparator = std::less<KeyType>,
 *           typename KeyEqualityChecker = std::equal_to<KeyType>,
 *           typename KeyHashFunc = std::hash<KeyType>,
 *           typename ValueEqualityChecker = std::equal_to<ValueType>,
 *           typename ValueHashFunc = std::hash<ValueType>>
 *
 * Explanation:
 *
 *  - KeyType: Key type of the map
 *
 *  - ValueType: Value type of the map. Note that it is possible
 *               that a single key is mapped to multiple values
 *
 *  - KeyComparator: "less than" relation comparator for KeyType
 *                   Returns true if "less than" relation holds
 *                   *** NOTE: THIS OBJECT DOES NOT NEED TO HAVE A DEFAULT
 *                   CONSTRUCTOR.
 *                   Please refer to main.cpp, class KeyComparator for more
 *                   information on how to define a proper key comparator
 *
 *  - KeyEqualityChecker: Equality checker for KeyType
 *                        Returns true if two keys are equal
 *
 *  - KeyHashFunc: Hashes KeyType into size_t. This is used in unordered_set
 *
 *  - ValueEqualityChecker: Equality checker for value type
 *                          Returns true for ValueTypes that are equal
 *
 *  - ValueHashFunc: Hashes ValueType into a size_t
 *                   This is used in unordered_set
 *
 * If not specified, then by default all arguments except the first two will
 * be set as the standard operator in C++ (i.e. the operator for primitive types
 * AND/OR overloaded operators for derived types)
 */
template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
          typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
          typename ValueEqualityChecker = std::equal_to<ValueType>>
class BPlusTree {
 public:
  explicit BPlusTree(KeyComparator p_key_cmp_obj = KeyComparator{},
                     KeyEqualityChecker p_key_eq_obj = KeyEqualityChecker{}, KeyHashFunc p_key_hash_obj = KeyHashFunc{},
                     ValueEqualityChecker p_value_eq_obj = ValueEqualityChecker{})
      : key_cmp_obj_{p_key_cmp_obj},
        key_eq_obj_{p_key_eq_obj},
        value_eq_obj_{p_value_eq_obj},
        epoch_(0),
        structure_size_{sizeof(LeafNode)} {
    root_ = static_cast<BaseNode *>(new LeafNode(this));
  }

  KeyComparator key_cmp_obj_;
  KeyEqualityChecker key_eq_obj_;
  ValueEqualityChecker value_eq_obj_;

  /*
   * KeyCmpLess() - Compare two keys for "less than" relation
   *
   * If key1 < key2 return true
   * If not return false
   *
   * NOTE: In older version of the implementation this might be defined
   * as the comparator to wrapped key type. However wrapped key has
   * been removed from the newest implementation, and this function
   * compares KeyType specified in template argument.
   */
  bool KeyCmpLess(const KeyType &key1, const KeyType &key2) const { return key_cmp_obj_(key1, key2); }

  /*
   * KeyCmpEqual() - Compare a pair of keys for equality
   *
   * This functions compares keys for equality relation
   */
  bool KeyCmpEqual(const KeyType &key1, const KeyType &key2) const { return key_eq_obj_(key1, key2); }

  /*
   * KeyCmpGreaterEqual() - Compare a pair of keys for >= relation
   *
   * It negates result of keyCmpLess()
   */
  bool KeyCmpGreaterEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpLess(key1, key2); }

  /*
   * KeyCmpGreater() - Compare a pair of keys for > relation
   *
   * It flips input for keyCmpLess()
   */
  bool KeyCmpGreater(const KeyType &key1, const KeyType &key2) const { return KeyCmpLess(key2, key1); }

  /*
   * KeyCmpLessEqual() - Compare a pair of keys for <= relation
   */
  bool KeyCmpLessEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpGreater(key1, key2); }

  std::atomic<uint64_t> epoch_;
  static const uint16_t BRANCH_FACTOR = 8;
  static const uint16_t LEAF_SIZE = 8;  // cannot ever be less than 3
  static const uint64_t WRITE_LOCK_MASK = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t NODE_TYPE_MASK = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t IS_DELETED_MASK = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t DELETE_EPOCH_MASK = ~(static_cast<uint64_t>(0xE) << 60);

  enum class NodeType : bool {
    LEAF = true,
    INNER_NODE = false,
  };

  class BaseNode {
   public:
    BaseNode(BPlusTree *tree, BPlusTree::NodeType t)
        : tree_(tree),
          info_(t == NodeType::LEAF ? NODE_TYPE_MASK : 0),
          size_(0),
          offset_(0),
          limit_(t == NodeType::LEAF ? LEAF_SIZE : BRANCH_FACTOR - 1) {}
    ~BaseNode() = default;

    BPlusTree *tree_;

    std::atomic<uint64_t> info_;
    std::atomic<uint16_t> size_, offset_, limit_;
    common::SharedLatch base_latch_;

    NodeType GetType() { return static_cast<NodeType>((info_ & NODE_TYPE_MASK) >> 62); }
  };

  class InnerNode : public BaseNode {
   public:
    explicit InnerNode(BPlusTree *tree) : BaseNode(tree, NodeType::INNER_NODE) {}
    ~InnerNode() = default;

    uint16_t FindMinChild(KeyType key) {
      uint16_t i;
      for (i = 0; i < this->size_.load(); i++)
        if (this->tree_->KeyCmpLessEqual(key, this->keys_[i])) return i;

      return i;
    }

    BaseNode *FindMaxChild(KeyType key) {
      for (uint16_t i = this->size_ - 1; i < this->size_; i--)
        if (this->tree_->KeyCmpGreaterEqual(key, keys_[i])) return children_[i + 1];

      return children_[0];
    }

    void InsertInner(KeyType key, BaseNode *child) {
      uint16_t j;
      for (j = 0; j < this->size_ && this->tree_->KeyCmpGreater(key, keys_[j]); j++) {
      }

      KeyType insertion_key = key;
      BaseNode *insertion_child = child;
      for (; j <= this->size_; j++) {
        std::swap(keys_[j], insertion_key);
        auto temp_child = insertion_child;
        insertion_child = children_[j + 1];
        children_[j + 1] = temp_child;
      }
      this->size_++;
    }

    std::atomic<BaseNode *> children_[BRANCH_FACTOR];
    KeyType keys_[BRANCH_FACTOR - 1];
  };

  class LeafNode : public BaseNode {
   public:
    static const uint64_t DELETE_MASK_SIZE = 64;

    explicit LeafNode(BPlusTree *tree) : BaseNode(tree, NodeType::LEAF), left_(nullptr), right_(nullptr) {}
    ~LeafNode() = default;

    bool ScanPredicate(KeyType key, std::function<bool(const ValueType)> predicate) {
      LeafNode *current_leaf = this;
      bool no_bigger_keys = true;
      while (current_leaf != nullptr && no_bigger_keys) {
        common::SharedLatch::ScopedSharedLatch l(&current_leaf->base_latch_);
        for (uint16_t i = 0; i < current_leaf->size_; i++) {
          if (current_leaf->tree_->KeyCmpEqual(key, current_leaf->keys_[i]) && predicate(current_leaf->values_[i])) {
            return true;
          }
          if (current_leaf->tree_->KeyCmpLess(key, current_leaf->keys_[i])) {
            no_bigger_keys = false;
          }
        }
        current_leaf = current_leaf->right_;
      }
      return false;
    }

    bool InsertLeaf(KeyType key, ValueType value, std::function<bool(const ValueType)> predicate,
                    bool *predicate_satisfied) {
      // look for deleted slot
      *predicate_satisfied = false;
      KeyType insert_key = key;
      ValueType insert_val = value;
      uint16_t i;
      for (i = 0; i < this->size_; i++) {
        if (this->tree_->KeyCmpEqual(key, keys_[i]) && predicate(values_[i])) {
          *predicate_satisfied = true;
          return false;
        }
        if (this->tree_->KeyCmpLess(key, keys_[i])) {
          break;
        }
      }

      if (i == this->size_ && this->right_ != nullptr &&
          (*predicate_satisfied = this->right_.load()->ScanPredicate(key, predicate))) {
        return false;
      }

      if (this->size_ >= this->limit_.load()) {
        return false;
      }

      for (; i < this->size_; i++) {
        std::swap(insert_key, keys_[i]);
        std::swap(insert_val, values_[i]);
      }

      keys_[i] = insert_key;
      values_[i] = insert_val;
      this->size_++;

      return true;
    }

    bool Remove(KeyType key, ValueType value) {
      common::SharedLatch::ScopedExclusiveLatch l(&this->base_latch_);
      if (this->size_ == 0) {
        return false;
      }
      if (this->tree_->KeyCmpEqual(key, keys_[this->size_ - 1]) &&
          this->tree_->ValueCmpEqual(value, values_[this->size_ - 1])) {
        this->size_--;
        return true;
      }

      KeyType last_key = keys_[this->size_ - 1];
      ValueType last_value = values_[this->size_ - 1];

      for (uint16_t i = 0; i < this->size_ - 1; i++)
        if (this->tree_->KeyCmpEqual(key, keys_[i]) && this->tree_->ValueCmpEqual(value, values_[i])) {
          keys_[i] = last_key;
          values_[i] = last_value;
          this->size_--;
          return true;
        }
      return false;
    }

    bool ScanRange(KeyType low, KeyType hi, std::vector<ValueType> *values) {
      // common::SharedLatch::ScopedSharedLatch l(&this->base_latch_);
      bool res = true;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (this->tree_->KeyCmpGreaterEqual(keys_[i], low) && this->tree_->KeyCmpLessEqual(keys_[i], hi))
          values->emplace_back(values_[i]);
        else if (this->tree_->KeyCmpGreater(keys_[i], hi))
          res = false;
      }
      return res;
    }

    std::atomic<LeafNode *> left_, right_;
    KeyType keys_[LEAF_SIZE] = {};
    ValueType values_[LEAF_SIZE] = {};
  };

  bool Insert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    common::SpinLatch::ScopedSpinLatch latch(&global_latch_);
    std::vector<InnerNode *> locked_nodes;
    std::vector<uint16_t> traversal_indices;
    BaseNode *n = this->root_;
    //    std::cout << "insert called" << std::endl;

    // Find minimum leaf that stores would store key, taking locks as we go down tree
    while (n->GetType() != NodeType::LEAF) {
      auto inner_n = static_cast<InnerNode *>(n);
      inner_n->base_latch_.LockExclusive();
      if (n->size_ < n->limit_) {
        for (InnerNode *node : locked_nodes) {
          node->base_latch_.Unlock();
        }
        locked_nodes.clear();
        traversal_indices.clear();
      }
      uint16_t child_index = inner_n->FindMinChild(key);
      n = inner_n->children_[child_index];
      locked_nodes.emplace_back(inner_n);
      traversal_indices.emplace_back(child_index);
    }

    // If leaf is not full, insert, unlock all parent nodes and return
    auto leaf = static_cast<LeafNode *>(n);
    common::SharedLatch::ScopedExclusiveLatch l(&leaf->base_latch_);
    //    std::cout << "Initial size: " << leaf->size_ << std::endl;
    if (leaf->InsertLeaf(key, val, predicate, predicate_satisfied) || *predicate_satisfied) {
      for (InnerNode *node : locked_nodes) {
        node->base_latch_.Unlock();
      }
      //      std::cout << "End size: " << leaf->size_ << std::endl;
      return !(*predicate_satisfied);
    }
    //    std::cout << "End size: " << leaf->size_ << std::endl;

    // Otherwise must split so create new leaf after sorting current leaf
    //    leaf->sort_leaf();
    structure_size_ += sizeof(LeafNode);
    auto new_leaf = new LeafNode(this);

    // Determine keys and values to stay in current leaf and copy others to new leaf
    uint16_t keep_in_leaf = leaf->size_ / 2;
    for (uint16_t i = keep_in_leaf; i < leaf->size_; i++) {
      new_leaf->keys_[i - keep_in_leaf] = leaf->keys_[i];
      new_leaf->values_[i - keep_in_leaf] = leaf->values_[i];
    }

    // Update sizes of current and newly created leaf
    new_leaf->size_ = leaf->size_ - keep_in_leaf;
    leaf->size_ = keep_in_leaf;

    // Insert given key and value into appropriate leaf
    if (KeyCmpLessEqual(key, new_leaf->keys_[0])) {
      leaf->InsertLeaf(key, val, predicate, predicate_satisfied);
      //      leaf->sort_leaf();
    } else {
      new_leaf->InsertLeaf(key, val, predicate, predicate_satisfied);
      //      new_leaf->sort_leaf();
    }

    // Update neighbor pointers for leaf nodes
    new_leaf->left_ = leaf;
    new_leaf->right_ = leaf->right_.load();
    leaf->right_ = new_leaf;
    if (LIKELY(new_leaf->right_ != nullptr)) {
      new_leaf->right_.load()->left_ = new_leaf;
    }

    // Determine pointer to new leaf to be pushed up to parent node
    KeyType new_key = new_leaf->keys_[0];

    // If split is on root (leaf)
    if (UNLIKELY(locked_nodes.empty())) {
      TERRIER_ASSERT(leaf == root_, "somehow we had to split a leaf without having to modify the parent");
      // Create new root and update attributes for new root and tree
      structure_size_ += sizeof(InnerNode);
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = leaf;
      new_root->children_[1] = new_leaf;
      new_root->size_ = 1;
      root_ = new_root;
      return true;
    }

    InnerNode *old_node = nullptr, *new_node = nullptr;
    auto new_child = static_cast<BaseNode *>(new_leaf);

    for (uint64_t i = locked_nodes.size() - 1;
         i < locked_nodes.size() && locked_nodes[i]->size_ == locked_nodes[i]->limit_; i--) {
      old_node = locked_nodes[i];
      uint16_t child_index = traversal_indices[i];
      structure_size_ += sizeof(InnerNode);
      new_node = new InnerNode(this);
      for (uint16_t node_index = child_index; node_index < old_node->size_; node_index++) {
        std::swap(new_key, old_node->keys_[node_index]);
        BaseNode *temp = new_child;
        new_child = old_node->children_[node_index + 1];
        old_node->children_[node_index + 1] = temp;
      }

      uint16_t lower_length = old_node->size_ / 2;
      new_node->size_ = old_node->size_ - lower_length;
      new_node->keys_[new_node->size_ - 1] = new_key;
      new_node->children_[new_node->size_] = new_child;

      for (uint16_t node_index = lower_length + 1; node_index < old_node->size_; node_index++) {
        new_node->keys_[node_index - lower_length - 1] = old_node->keys_[node_index];
        new_node->children_[node_index - lower_length] = old_node->children_[node_index + 1].load();
      }
      new_node->children_[0] = old_node->children_[lower_length + 1].load();
      old_node->size_ = lower_length;
      new_key = old_node->keys_[lower_length];
      new_child = static_cast<BaseNode *>(new_node);
    }

    // Root is being split so must create new root node
    if (old_node == root_) {
      structure_size_ += sizeof(InnerNode);
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = old_node;
      new_root->children_[1] = new_node;
      new_root->size_ = 1;
      root_ = new_root;
    } else {
      TERRIER_ASSERT(locked_nodes[0]->size_ != locked_nodes[0]->limit_, "top node in split was full but not the root");

      locked_nodes[0]->InsertInner(new_key, new_child);
    }

    // Unlock all latches incurred on insert
    for (InnerNode *node : locked_nodes) {
      node->base_latch_.Unlock();
    }
    return true;
  }

  void ScanKey(KeyType key, std::vector<ValueType> *values) {
    common::SpinLatch::ScopedSpinLatch latch(&global_latch_);
    InnerNode *parent;
    BaseNode *n = this->root_;
    while (n->GetType() != NodeType::LEAF) {
      auto *inner_n = static_cast<InnerNode *>(n);
      inner_n->base_latch_.LockShared();
      parent = static_cast<InnerNode *>(n);
      uint16_t n_index = inner_n->FindMinChild(key);
      n = inner_n->children_[n_index];
      parent->base_latch_.Unlock();
    }
    auto leaf = static_cast<LeafNode *>(n);
    LeafNode *sibling;
    leaf->base_latch_.LockShared();
    while (true) {
      if (!leaf->ScanRange(key, key, values)) {
        leaf->base_latch_.Unlock();
        break;
      }
      sibling = leaf;
      leaf = leaf->right_;
      if (leaf == nullptr) {
        sibling->base_latch_.Unlock();
        break;
      }
      leaf->base_latch_.LockShared();
      sibling->base_latch_.Unlock();
    }
  }

  std::atomic<BaseNode *> root_;
  std::atomic<uint64_t> structure_size_;
  common::SpinLatch global_latch_;
};

}  // namespace terrier::storage::index
