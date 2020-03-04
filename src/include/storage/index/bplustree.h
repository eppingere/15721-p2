#pragma once

#include <common/shared_latch.h>

#include <execution/util/execution_common.h>
#include <storage/storage_defs.h>
#include <functional>

#include "common/macros.h"

namespace terrier::storage::index {

template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
          typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
          typename ValueEqualityChecker = std::equal_to<ValueType>>
class BPlusTree {
 public:
  BPlusTree(KeyComparator p_key_cmp_obj = KeyComparator{}, KeyEqualityChecker p_key_eq_obj = KeyEqualityChecker{},
            KeyHashFunc p_key_hash_obj = KeyHashFunc{}, ValueEqualityChecker p_value_eq_obj = ValueEqualityChecker{})
      : key_cmp_obj{p_key_cmp_obj},
        key_eq_obj{p_key_eq_obj},
        value_eq_obj{p_value_eq_obj},
        structure_size_{sizeof(LeafNode)} {
    root_ = static_cast<BaseNode *>(new LeafNode(this));
  };

  KeyComparator key_cmp_obj;
  KeyEqualityChecker key_eq_obj;
  ValueEqualityChecker value_eq_obj;

  inline bool KeyCmpLess(const KeyType &key1, const KeyType &key2) const { return key_cmp_obj(key1, key2); }
  inline bool KeyCmpEqual(const KeyType &key1, const KeyType &key2) const { return key_eq_obj(key1, key2); }
  inline bool KeyCmpGreaterEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpLess(key1, key2); }
  inline bool KeyCmpGreater(const KeyType &key1, const KeyType &key2) const { return KeyCmpLess(key2, key1); }
  inline bool KeyCmpLessEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpGreater(key1, key2); }
  inline bool ValueCmpEqual(const ValueType &v1, const ValueType &v2) { return value_eq_obj(v1, v2); }

  static std::atomic<uint64_t> epoch;
  static const short branch_factor_ = 8;
  static const short leaf_size_ = 8;  // cant every be less than 3
  static const uint64_t write_lock_mask_ = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t node_type_mask_ = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t is_deleted_mask_ = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t delete_epoch_mask_ = ~(static_cast<uint64_t>(0xE) << 60);

  enum class NodeType : bool {
    LEAF = true,
    INNER_NODE = false,
  };

  class BaseNode {
   public:
    BaseNode(BPlusTree *tree, BPlusTree::NodeType t)
        : tree_(tree),
          info_(t == NodeType::LEAF ? node_type_mask_ : 0),
          size_(0),
          offset_(0),
          limit_(t == NodeType::LEAF ? leaf_size_ : branch_factor_ - 1){};
    ~BaseNode() = default;

    BPlusTree *tree_;

    std::atomic<uint64_t> info_;
    std::atomic<uint16_t> size_, offset_, limit_;
    common::SharedLatch base_latch_;

    inline void write_lock() {
      while (true) {
        uint64_t old_value = info_.load();
        uint64_t not_locked = old_value | (~write_lock_mask_);
        uint64_t locked = old_value | write_lock_mask_;
        if (!info_.compare_exchange_strong(not_locked, locked)) {
          continue;
        }
        break;
      }
    };

    inline void write_unlock() {
      TERRIER_ASSERT(info_.load() & write_lock_mask_, "unlock called on unlocked inner node");
      while (true) {
        uint64_t old_meta_data = info_.load();
        if (!info_.compare_exchange_strong(old_meta_data, old_meta_data & (~write_lock_mask_))) {
          continue;
        }
        break;
      }
    };

    inline void mark_delete() {
      TERRIER_ASSERT(!(info_.load() & is_deleted_mask_), "tried to mark as deleted already deleted node");
      while (true) {
        uint64_t old_meta_data = info_.load() & (~delete_epoch_mask_);
        uint64_t e = epoch.load() & delete_epoch_mask_;

        if (!info_.compare_exchange_strong(old_meta_data, old_meta_data | is_deleted_mask_ | e)) {
          continue;
        }
        break;
      }
    }

    inline uint64_t get_delete_epoch() {
      TERRIER_ASSERT(info_.load() & is_deleted_mask_, "got delete epoch of non deleted node");
      return info_.load() & delete_epoch_mask_;
    }

    inline NodeType get_type() { return static_cast<NodeType>((info_ & node_type_mask_) >> 62); }
  };

  class InnerNode : public BaseNode {
   public:
    friend class BplusTree;
    InnerNode(BPlusTree *tree) : BaseNode(tree, NodeType::INNER_NODE){};
    ~InnerNode() = default;

    uint16_t findMinChild(KeyType key) {
      uint16_t i;
      for (i = 0; i < this->size_.load(); i++)
        if (this->tree_->KeyCmpLess(key, this->keys_[i])) return i;

      return i;
    }

    BaseNode *findMaxChild(KeyType key) {
      for (uint16_t i = 0; i < this->size_; i++)
        if (this->tree_->KeyCmpGreaterEqual(key, keys_[i])) return children_[i + 1];

      return children_[0];
    }

    void insert_inner(KeyType key, BaseNode *child) {
      uint16_t j;
      for (j = 0; j < this->size_ && this->tree_->KeyCmpLess(key, keys_[j]); j++) {
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

    std::atomic<BaseNode *> children_[branch_factor_];
    KeyType keys_[branch_factor_ - 1];
  };

  class LeafNode : public BaseNode {
   public:
    static const uint64_t delete_mask_size_ = 64;

    LeafNode(BPlusTree *tree) : BaseNode(tree, NodeType::LEAF), left_(nullptr), right_(nullptr){};
    ~LeafNode() = default;

    bool insert(KeyType key, ValueType value, std::function<bool(const ValueType)> predicate,
                bool *predicate_satisfied) {
      // look for deleted slot
      if (this->size_ >= this->limit_.load()) {
        return false;
      }

      KeyType insert_key = key;
      ValueType insert_val = value;
      uint16_t i;
      *predicate_satisfied = false;
      for (i = 0; i < this->size_; i++) {
        if (this->tree_->KeyCmpEqual(key, keys_[i]) && predicate(values_[i])) {
          *predicate_satisfied = true;
          return false;
        } else if (this->tree_->KeyCmpLess(key, keys_[i])) {
          break;
        }
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

    //    bool compare_pair (std::pair<KeyType,ValueType> a,  std::pair<KeyType,ValueType> b) {
    //      return this->tree_->KeyCmpLess(a.first, b.first);
    //    }

    //    void sort_leaf() {
    //      uint16_t num_nodes = this->size_.load();
    //      auto keys = keys_;
    //      auto values = values_;
    //      std::pair<KeyType, ValueType> pairs[num_nodes];
    //      for (uint16_t i = 0; i < num_nodes; i++) {
    //        pairs[i].first = keys[i];
    //        pairs[i].second = values[i];
    //      }
    //
    //      sort(pairs, pairs + num_nodes, compare_pair);
    //
    //      for (uint16_t i = 0; i < num_nodes; i++) {
    //        keys[i] = pairs[i].first;
    //        values[i] = pairs[i].second;
    //      }
    //    }

    bool scan_range(KeyType low, KeyType hi, std::vector<ValueType> *values) {
      common::SharedLatch::ScopedSharedLatch l(&this->base_latch_);
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
    KeyType keys_[leaf_size_];
    ValueType values_[leaf_size_];
  };

  bool Insert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    std::vector<InnerNode *> locked_nodes;
    std::vector<uint16_t> traversal_indices;
    BaseNode *n = this->root_;

    // Find minimum leaf that stores would store key, taking locks as we go down tree
    while (n->get_type() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      inner_n->base_latch_.LockExclusive();
      if (n->size_ < n->limit_) {
        for (InnerNode *node : locked_nodes) {
          node->base_latch_.Unlock();
        }
        locked_nodes.clear();
        traversal_indices.clear();
      }
      uint16_t child_index = inner_n->findMinChild(key);
      n = inner_n->children_[child_index];
      locked_nodes.emplace_back(inner_n);
      traversal_indices.emplace_back(child_index);
    }

    // If leaf is not full, insert, unlock all parent nodes and return
    LeafNode *leaf = static_cast<LeafNode *>(n);
    common::SharedLatch::ScopedExclusiveLatch l(&leaf->base_latch_);
    if (leaf->insert(key, val, predicate, predicate_satisfied) || predicate_satisfied) {
      for (InnerNode *node : locked_nodes) {
        node->base_latch_.Unlock();
      }
      return !(*predicate_satisfied);
    }

    // Otherwise must split so create new leaf after sorting current leaf
    //    leaf->sort_leaf();
    structure_size_ += sizeof(LeafNode);
    LeafNode *new_leaf = new LeafNode(this);

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
    if (KeyCmpLess(key, new_leaf->keys_[0])) {
      leaf->insert(key, val, predicate, predicate_satisfied);
      //      leaf->sort_leaf();
    } else {
      new_leaf->insert(key, val, predicate, predicate_satisfied);
      //      new_leaf->sort_leaf();
    }

    // Update neighbor pointers for leaf nodes
    new_leaf->left_ = leaf;
    new_leaf->right_ = leaf->right_.load();
    leaf->right_ = new_leaf;
    if (UNLIKELY(new_leaf->right_ != nullptr)) {
      new_leaf->right_.load()->left_ = new_leaf;
    }

    // Determine pointer to new leaf to be pushed up to parent node
    KeyType new_key = new_leaf->keys_[0];

    // If split is on root (leaf)
    if (UNLIKELY(locked_nodes.size() == 0)) {
      TERRIER_ASSERT(leaf == root_, "somehow we had to split a leaf without having to modify the parent");
      // Create new root and update attributes for new root and tree
      structure_size_ += sizeof(InnerNode);
      InnerNode *new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = leaf;
      new_root->children_[1] = new_leaf;
      root_.load()->size_ = 1;
      root_ = new_root;
      return true;
    }

    InnerNode *old_node, *new_node;
    BaseNode *new_child = static_cast<BaseNode *>(new_leaf);

    for (uint64_t i = locked_nodes.size() - 1; i > 0; i--) {
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
    if (locked_nodes[0] == root_ && locked_nodes[0]->size_ == locked_nodes[0]->limit_) {
      structure_size_ += sizeof(InnerNode);
      InnerNode *new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = old_node;
      new_root->children_[1] = new_node;
      root_.load()->size_ = 1;
      root_ = new_root;
    } else {
      TERRIER_ASSERT(locked_nodes[0]->size_ != locked_nodes[0]->limit_, "top node in split was full but not the root");

      locked_nodes[0]->insert_inner(new_key, new_child);
    }

    // Unlock all latches incurred on insert
    for (InnerNode *node : locked_nodes) {
      node->base_latch_.Unlock();
    }
    return true;
  }

  std::vector<ValueType> ScanKey(KeyType key) {
    std::vector<ValueType> value_list;
    InnerNode *parent;
    BaseNode *n = this->root_;
    while (n->get_type() != NodeType::LEAF) {
      auto *inner_n = static_cast<InnerNode *>(n);
      inner_n->base_latch_.LockShared();
      parent = n;
      n = inner_n->findMinChild(key);
      parent->base_latch_.Unlock();
    }
    auto *leaf = static_cast<LeafNode *>(n);
    while(leaf != NULL && leaf->scan_range(key, key, &value_list)) {
      leaf = leaf->right_;
    }
    return value_list;
  }

  std::atomic<BaseNode *> root_;
  common::SharedLatch root_latch_;
  std::atomic<uint64_t> structure_size_;

  // TODO: Check if bool or bool (*) is correct type of predicate
  LeafNode *FindMin(KeyType key) {}
  LeafNode *FindMax(KeyType key) {}
  void FindRange(KeyType min_key, KeyType max_key, bool is_increasing, std::vector<ValueType> *results);
};

}  // namespace terrier::storage::index