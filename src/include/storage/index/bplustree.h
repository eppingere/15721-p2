#pragma once

#include <common/shared_latch.h>

#include <functional>
#include <storage/storage_defs.h>
#include <execution/util/execution_common.h>

#include "common/macros.h"

namespace terrier::storage::index {

template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
    typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
    typename ValueEqualityChecker = std::equal_to<ValueType>>
class BPlusTree {
 public:
  BPlusTree(KeyComparator p_key_cmp_obj = KeyComparator{},
            KeyEqualityChecker p_key_eq_obj = KeyEqualityChecker{}, KeyHashFunc p_key_hash_obj = KeyHashFunc{},
            ValueEqualityChecker p_value_eq_obj = ValueEqualityChecker{})
      : root_{static_cast<BaseNode *>(new LeafNode)}, structure_size_(sizeof(LeafNode)) {};

  static std::atomic<uint64_t> epoch;
  static const short branch_factor_ = 8;
  static const short leaf_size_ = 8; // cant every be less than 3
  static const uint64_t write_lock_mask_ = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t node_type_mask_ = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t is_deleted_mask_ = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t delete_epoch_mask_ = ~(static_cast<uint64_t>(0xE) << 60);

  enum class NodeType: bool {
    LEAF = true,
    INNER_NODE = false,
  };

  class BaseNode {
   public:
    BaseNode(BPlusTree::NodeType t) :
        info_(t == NodeType::LEAF ? node_type_mask_ : 0),
        size_(0),
        offset_(0),
        limit_(t == NodeType::LEAF ? leaf_size_ : branch_factor_ - 1) {};
    ~BaseNode()=default;

    std::atomic<uint64_t> info_;
    std::atomic<uint16_t> size_, offset_, limit_;
    common::SharedLatch base_latch_;

    inline void write_lock() {
      while(true) {
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
      TERRIER_ASSERT(!(info_.load() & is_deleted_mask_),
                     "tried to mark as deleted already deleted node");
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

    inline NodeType get_type() { return static_cast<NodeType>(info_ & node_type_mask_); }
  };

  class InnerNode : public BaseNode {
   public:
    InnerNode() : BaseNode(NodeType::INNER_NODE) {};
    ~InnerNode()=default;


    BaseNode* findMinChild(KeyType key) {
      uint16_t i;
      for (i = 0; i < this->size_.load(); i++)
        if (KeyCmpLess(key, keys_[i]))
          return children_[i];

      return children_[i];
    }

    BaseNode* findMaxChild(KeyType key) {
      for (uint16_t i = 0; i < this->size_; i++)
        if (KeyCmpGreaterEqual(key, keys_[i]))
          return children_[i + 1];

      return children_[0];
    }

    void insert_inner(KeyType key, BaseNode* child) {
      uint16_t j = 0;
      while (j < this->size_ && KeyCmpLessEqual(keys_[j], key)) {
        j++;
      }

      KeyType insertion_key = key;
      BaseNode* insertion_child = child;
      for (; j <= this->size_; j++) {
        std::swap(keys_[j], insertion_key);
        std::swap(children_[j + 1], child);
      }
      this->size_++;
    }

    std::atomic<BaseNode *> children_[branch_factor_];
    KeyType keys_[branch_factor_ - 1];
  };

  class LeafNode : public BaseNode {
   public:
    static const uint64_t delete_mask_size_ = 64;

    LeafNode() : BaseNode(NodeType::LEAF), left_(nullptr), right_(nullptr) {};
    ~LeafNode()=default;

    bool insert(KeyType key, ValueType value) {
      // look for deleted slot
      if (this->size_ >= this->limit_.load()) {
        return false;
      }

      keys_[this->size_] = key;
      values_[this->size_] = value;
      this->size_++;

      return true;
    }

    bool Remove(KeyType key, ValueType value) {
      common::SharedLatch::ScopedExclusiveLatch l(&this->base_latch_);
      if (this->size_ == 0) {
        return false;
      }
      if (KeyCmpEqual(key, keys_[this->size_ - 1]) && ValueCmpEqual(value, values_[this->size_ - 1])) {
        this->size_--;
        return true;
      }

      KeyType last_key = keys_[this->size_ - 1];
      ValueType last_value = values_[this->size_ - 1];

      for (uint16_t i = 0; i < this->size_ - 1; i++)
        if (KeyCmpEqual(key, keys_[i]) && ValueCmpEqual(value, values_[i])) {
          keys_[i] = last_key;
          values_[i] = last_value;
          this->size_--;
          return true;
        }
      return false;
    }

    void sort_leaf() {
      uint16_t num_nodes = this->size_.load();
      auto keys = keys_;
      auto values = values_;
      std::pair<KeyType, ValueType> pairs[num_nodes];
      for (uint16_t i = 0; i < num_nodes; i++) {
        pairs[i].first = keys[i];
        pairs[i].second = values[i];
      }

      sort(pairs, pairs + num_nodes);

      for (uint16_t i = 0; i < num_nodes; i++) {
        keys[i] = pairs[i].first;
        values[i] = pairs[i].second;
      }
    }

    bool scan_range(KeyType low, KeyType hi, std::vector<ValueType> *values) {
      common::SharedLatch::ScopedSharedLatch l(&this->base_latch_);
      bool res = true;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (KeyCmpGreaterEqual(keys_[i], low) && KeyCmpLessEqual(keys_[i], hi))
          values->emplace_back(values_[i]);
        else if (KeyCmpGreater(keys_[i], hi))
          res = false;
      }
      return res;
    }

    std::atomic<LeafNode*> left_, right_;
    KeyType keys_[leaf_size_];
    ValueType values_[leaf_size_];
  };

  void Insert(KeyType key, ValueType val) {
    std::vector<InnerNode *> locked_nodes;
    BaseNode *n = this->root_;
    while (n->get_type() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      inner_n->base_latch_.LockExclusive();
      if (n->size_ < n->limit_) {
        for (InnerNode *node : locked_nodes) {
          node->base_latch_.Unlock();
        }
        locked_nodes.clear();
      }
      locked_nodes.emplace_back(inner_n);
      n = inner_n->findMinChild(key);
    }

    LeafNode *leaf = static_cast<LeafNode *>(n);
    common::SharedLatch::ScopedExclusiveLatch l(&leaf->base_latch_);
    if (leaf->insert(key, val)) {
      for (InnerNode *node : locked_nodes) {
        node->base_latch_.Unlock();
        return;
      }
    }

    leaf->sort_leaf();
    structure_size_ += sizeof(LeafNode);

    LeafNode* new_leaf = new LeafNode();
    uint16_t keep_in_leaf = leaf->size_ / 2;
    for (uint16_t i = keep_in_leaf; i < leaf->size_; i++) {
      new_leaf->keys_[i - keep_in_leaf] = leaf->keys_[i];
      new_leaf->values_[i - keep_in_leaf] = leaf->values_[i];
    }
    new_leaf->size_ = leaf->size_ - keep_in_leaf;
    leaf->size_ = keep_in_leaf;

    if (KeyCmpLess(key, new_leaf->keys_[0])) {
      leaf->insert(key, val);
    } else {
      new_leaf->insert(key, val);
    }

    new_leaf->left_ = leaf;
    new_leaf->right_ = leaf->right_;
    leaf->right_ = new_leaf;
    if (UNLIKELY(new_leaf->right_ != nullptr)) {
      new_leaf->right_->left_ = new_leaf;
    }

    KeyType new_key = new_leaf->keys_[0];

    if (UNLIKELY(locked_nodes.size() == 0)) {
      TERRIER_ASSERT(leaf == root_, "somehow we had to split a leaf without having to modify the parent");
      structure_size_ += sizeof(InnerNode);
      InnerNode* new_root = new InnerNode();
      new_root->keys_[0] = new_key;
      new_root->children_[0] = leaf;
      new_root->children_[1] = new_leaf;
      root_->size_ = 1;
      root_ = new_root;
      return;
    }

    int32_t i = locked_nodes.size() - 1;
    InnerNode* old_node = locked_nodes[i];
    BaseNode* child_node = static_cast<BaseNode *>(new_leaf);
    InnerNode* new_node;
    bool splitting_root = locked_nodes[0] == root_ && locked_nodes[0]->size_ == locked_nodes[0]->limit_;

    while (i > 0 && old_node->size_ == old_node->limit_) {
      structure_size_ += sizeof(InnerNode);

      new_node = new InnerNode();
      uint16_t keep_in_node = old_node->size_ / 2;
      for (uint16_t j = keep_in_node; j < old_node->size_; j++) {
        new_node->keys_[j - keep_in_node] = old_node->keys_[j];
        new_node->values_[j - keep_in_node] = old_node->values_[j];
      }
      old_node->size_ = keep_in_node;

      if (KeyCmpLess(new_key, new_node->keys_[0])) {
        old_node->insert(new_key, val);
      } else {
        new_node->insert(new_key, val);
      }

      i--;
    }

    if (splitting_root) {
      structure_size_ += sizeof(InnerNode);
      InnerNode* new_root = new InnerNode();
      new_root->keys_[0] = new_key;
      new_root->children_[0] = old_node;
      new_root->children_[1] = new_node;
      root_->size_ = 1;
      root_ = new_root;
    } else {
      TERRIER_ASSERT(locked_nodes[0]->size_ != locked_nodes[0]->limit_,
                     "top node in split was full but not the root");

      locked_nodes[0]->insert_inner(new_key, child_node);
    }

  }

  inline bool KeyCmpLess(const KeyType &key1, const KeyType &key2) const { return key_cmp_obj(key1, key2); }

  inline bool KeyCmpEqual(const KeyType &key1, const KeyType &key2) const { return key_eq_obj(key1, key2); }

  inline bool KeyCmpGreaterEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpLess(key1, key2); }

  inline bool KeyCmpGreater(const KeyType &key1, const KeyType &key2) const { return KeyCmpLess(key2, key1); }

  inline bool KeyCmpLessEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpGreater(key1, key2); }

  inline bool ValueCmpEqual(const ValueType &v1, const ValueType &v2) { return value_eq_obj(v1, v2); }

  std::atomic<BaseNode *> root_;
  std::atomic<uint64_t> structure_size_;

  // TODO: Check if bool or bool (*) is correct type of predicate
  bool InsertIf(std::function<bool (const ValueType)> predicate, KeyType key, ValueType value);
  LeafNode* FindMin(KeyType key);
  LeafNode* FindMax(KeyType key);
  void FindRange(KeyType min_key, KeyType max_key, bool is_increasing, std::vector <ValueType> *results);
};

}  // namespace terrier::storage::index