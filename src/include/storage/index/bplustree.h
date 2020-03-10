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
  /*
   * Constructor - Set up initial environment for BwTree
   *
   * Any tree instance will start with an empty leaf node as the root, which will then be filled.
   *
   * Some properties of the tree should be specified in the argument.
   */
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

  // Key comparator, and key and value equality checker
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

  // Tunable parameters
  static const uint64_t BITS_IN_UINT64 = 8 * sizeof(uint64_t);
  static const uint16_t BRANCH_FACTOR = 20;
  static const uint16_t LEAF_SIZE = 20;  // cannot ever be less than 3
  static const uint64_t WRITE_LOCK_MASK = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t NODE_TYPE_MASK = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t IS_DELETED_MASK = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t DELETE_EPOCH_MASK = ~(static_cast<uint64_t>(0xE) << 60);

  // Node types
  enum class NodeType : bool {
    LEAF = true,
    INNER_NODE = false,
  };

  template <class T>
  class NodeAllocator {
    static const uint64_t ALLOCATOR_ARRAY_SIZE = 64;

    class NodeAllocatorArray {
      NodeAllocatorArray(std::function<T(BPlusTree)> constructor) : allocated_masks_({}), array_(new T[ALLOCATOR_ARRAY_SIZE]) {
        for (uint64_t i = 0; i < ALLOCATOR_ARRAY_SIZE; i++) {
          array_[i] = constructor(this);
        }
      }

      bool Allocate(uint64_t i) {
        while (true) {
          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
          uint64_t new_value =
              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
          uint64_t available = old_value >> (i % BITS_IN_UINT64);
          if (!static_cast<bool>(available & static_cast<uint64_t>(0x1))) return false;
          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
          return true;
        }
      }

      bool IsAvailable(uint64_t i) {
        uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
        return !static_cast<bool>((old_value >> (i % BITS_IN_UINT64)) & static_cast<uint64_t>(0x1));
      }

      void Reclaim(uint64_t i) {
        while (true) {
          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
          uint64_t new_value =
              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
          return;
        }
      }

      std::atomic<uint64_t> allocated_masks_[(ALLOCATOR_ARRAY_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64];
      T array_[ALLOCATOR_ARRAY_SIZE];
    };

    NodeAllocator(uint64_t starting_size, std::function<T(BPlusTree)> constructor) : constructor_(constructor),
    resizing_(false),
    size_((starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE),
    table_(new NodeAllocatorArray*[(starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE]) {
      for (uint64_t i = 0; i < (starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE; i++) {
        table_[i] = new NodeAllocatorArray(constructor);
      }
    }

    T* NewNode() {
      do {
        for (uint64_t i = 0; i < size_; i++) {
          uint64_t j;
          for (j = 0; j < ALLOCATOR_ARRAY_SIZE && !table_[i]->Allocate(j); j++) {
          }
          if (j == ALLOCATOR_ARRAY_SIZE) return table_[i]->array_ + j;
        }
      } while (resizing_);

      while (!resizing_.compare_exchange_strong(false, true)) {}
      NodeAllocatorArray ** new_table = new NodeAllocatorArray*[2 * size_];
      for (uint64_t i = 0; i < size_; i++) {
        new_table[i] = table_[i];
      }

      for (uint64_t i = size_; i < 2 * size_; i++) {
        new_table[i] = new NodeAllocatorArray(constructor_);
      }

      table_ = new_table;
      resizing_ = false;
    }

    std::function<T(BPlusTree)> constructor_;
    std::atomic<bool> resizing_;
    std::atomic<uint64_t> size_;
    std::atomic<NodeAllocatorArray **> table_;
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
    std::atomic<bool> write_latch = false;
    std::atomic<bool> deleted_ = false;
    std::atomic<uint64_t> deleted_epoch_ = 0;
    std::atomic<uint64_t> info_;
    std::atomic<uint16_t> size_, offset_, limit_;

    NodeType GetType() { return static_cast<NodeType>((info_ & NODE_TYPE_MASK) >> 62); }

    void MarkDeleted() {
      deleted_ = true;
      deleted_epoch_ = tree_->epoch_.load();
    }

    void Reallocate() {
      write_latch = false;
      deleted_ = false;
      deleted_epoch_ = 0;
      size_ = 0;
      offset_ = 0;
    }

    void GetWriteLatch() {
      bool t = true;
      bool f = false;
      while (!write_latch.compare_exchange_strong(f, t)) {}
    }
    void ReleaseWriteLatch() { write_latch = false; }

    class ScopedWriteLatch {
     public:
      ScopedWriteLatch(BaseNode* node) : node_(node) {
        node_->GetWriteLatch();
      }
      ~ScopedWriteLatch() {
        node_->ReleaseWriteLatch();
      }
      BaseNode* node_;
    };
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

    InnerNode* Merge() {
      BaseNode::ScopedWriteLatch (this);

    }

    std::atomic<BaseNode *> children_[BRANCH_FACTOR];
    KeyType keys_[BRANCH_FACTOR - 1];
  };

  class LeafNode : public BaseNode {
   public:
    explicit LeafNode(BPlusTree *tree) : BaseNode(tree, NodeType::LEAF), left_(nullptr), right_(nullptr) {}
    ~LeafNode() = default;

    bool ScanPredicate(KeyType key, std::function<bool(const ValueType)> predicate) {
      LeafNode *current_leaf = this;
      bool no_bigger_keys = true;
      while (current_leaf != nullptr && no_bigger_keys) {
        for (uint16_t i = 0; i < current_leaf->size_; i++) {
          if (current_leaf->IsReadable(i)) {
            if (current_leaf->tree_->KeyCmpEqual(key, current_leaf->keys_[i]) && predicate(current_leaf->values_[i])) {
              return true;
            }
            if (current_leaf->tree_->KeyCmpLess(key, current_leaf->keys_[i])) {
              no_bigger_keys = false;
            }
          }
        }
        current_leaf = current_leaf->right_;
      }
      return false;
    }

    bool InsertLeaf(KeyType key, ValueType value, std::function<bool(const ValueType)> predicate,
                    bool *predicate_satisfied) {
      typename BaseNode::ScopedWriteLatch l(this);
      if (UNLIKELY(this->deleted_)) {
        return false;
      }

      *predicate_satisfied = false;
      bool saw_bigger = false;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (IsReadable(i)) {
          if (this->tree_->KeyCmpEqual(key, keys_[i]) && predicate(values_[i])) {
            *predicate_satisfied = true;
            return false;
          } else if (this->tree_->KeyCmpGreater(keys_[i], key)) {
            saw_bigger = true;
          }
        }
      }

      LeafNode *right = right_;
      if (!saw_bigger && right != nullptr && (*predicate_satisfied = right->ScanPredicate(key, predicate))) {
        return false;
      }

      // look for deleted slot
      uint16_t i;
      for (i = 0; i < this->size_ && IsReadable(i); i++) {}
      if (i != this->size_) {
        keys_[i] = key;
        values_[i] = value;
        UnmarkTombStone(i);
        return true;
      }

      if (this->size_ >= this->limit_) {
        return false;
      }

      keys_[this->size_] = key;
      values_[this->size_] = value;
      this->size_++;
      return true;
    }

    bool ScanRange(KeyType low, KeyType *hi, std::vector<ValueType> *values) {
      bool res = true;
      std::vector<std::pair<KeyType, ValueType>> temp_values;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (IsReadable(i)) {
          if (this->tree_->KeyCmpLessEqual(low, keys_[i]) &&
              (hi == nullptr || this->tree_->KeyCmpLessEqual(keys_[i], *hi))) {
            temp_values.emplace_back(std::pair<KeyType, ValueType>(keys_[i], values_[i]));
          } else if (hi == nullptr || this->tree_->KeyCmpGreater(keys_[i], *hi)) {
            res = false;
          }
        }
      }

      sort(temp_values.begin(), temp_values.end(),  [&] (std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
        return this->tree_->KeyCmpLess(a.first, b.first);
      });

      for (uint16_t i = 0; i < temp_values.size(); i++) {
          values->emplace_back(temp_values[i].second);
      }

      return res;
    }

    // the tomb stoning could be made faster with better bit stuff
    bool IsReadable(uint16_t  i) {
      return !static_cast<bool>((tomb_stones_[i / BITS_IN_UINT64] >> (i % BITS_IN_UINT64)) & static_cast<uint64_t>(0x1));
    }

    void MarkTombStone(uint16_t i) {
      while (true) {
        uint64_t old_value = tomb_stones_[i / BITS_IN_UINT64];
        uint16_t new_value = old_value | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
        if (!tomb_stones_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
        return;
      }
    }

    void UnmarkTombStone(uint16_t i) {
      while (true) {
        uint64_t old_value = tomb_stones_[i / BITS_IN_UINT64];
        uint16_t new_value = old_value & (~(static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64)));
        if (!tomb_stones_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
        return;
      }
    }

    // at least as many bits as key value pairs in leaf
    std::atomic<uint64_t> tomb_stones_[(LEAF_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64] = {};
    std::atomic<LeafNode *> left_, right_;
    KeyType keys_[LEAF_SIZE] = {};
    ValueType values_[LEAF_SIZE] = {};
  };

  inline bool OptimisticInsert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    return FindMinLeaf(key)->InsertLeaf(key, val, predicate, predicate_satisfied);
  }

  bool Insert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    if (OptimisticInsert(predicate, key, val, predicate_satisfied)) {
      return true;
    }

    std::vector<InnerNode *> locked_nodes;
    std::vector<uint16_t> traversal_indices;
    uint64_t locked_index = 0; // everything in locked_nodes from this index onwards is locked

    LatchRoot();
    bool holds_tree_latch = true;
    BaseNode *n = this->root_;

    // Find minimum leaf that stores would store key, taking locks as we go down tree
    while (n->GetType() != NodeType::LEAF) {
      auto inner_n = static_cast<InnerNode *>(n);
      n->GetWriteLatch();
      if (n->size_ < n->limit_) {
        if (holds_tree_latch && locked_index != 0) {
          UnlatchRoot();
          holds_tree_latch = false;
        }
        for (; !locked_nodes.empty() && locked_index < locked_nodes.size() - 1; locked_index++) {
          locked_nodes[locked_index]->ReleaseWriteLatch();
        }
      }
      uint16_t child_index = inner_n->FindMinChild(key);
      n = inner_n->children_[child_index];
      locked_nodes.emplace_back(inner_n);
      traversal_indices.emplace_back(child_index);
    }

    // If leaf is not full, insert, unlock all parent nodes and return
    auto leaf = static_cast<LeafNode *>(n);
    if (leaf->InsertLeaf(key, val, predicate, predicate_satisfied) || *predicate_satisfied) {
      if (holds_tree_latch) {
        UnlatchRoot();
      }
      for (; locked_index < locked_nodes.size(); locked_index++) {
        locked_nodes[locked_index]->ReleaseWriteLatch();
      }
      return !(*predicate_satisfied);
    }

    // returned false because of no tomb stones;

    LeafNode *left, *right;
    while (true) {
      left = leaf->left_;
      right = leaf->right_;
      if (LIKELY(left != nullptr)) {
        left->GetWriteLatch();
        if (UNLIKELY(left != leaf->left_)) {
          left->ReleaseWriteLatch();
          continue;
        }
      }
      leaf->GetWriteLatch();
      if (LIKELY(right != nullptr)) {
        right->GetWriteLatch();
        if (UNLIKELY(right != leaf->right_)) {
          if (LIKELY(left != nullptr)) left->ReleaseWriteLatch();
          leaf->ReleaseWriteLatch();
          right->ReleaseWriteLatch();
          continue;
        }
      }
      break;
    }

    // Otherwise must split so create new leaf
    LeafNode* new_leaf_right = new LeafNode(this);
    LeafNode* new_leaf_left = new LeafNode(this);

    std::vector<std::pair<KeyType, ValueType>> kvps;
    kvps.emplace_back(std::pair<KeyType, ValueType>(key, val));
    for (uint16_t i = 0; i < leaf->size_; i++) {
      kvps.emplace_back(std::pair<KeyType, ValueType>(leaf->keys_[i], leaf->values_[i]));
    }

    sort(kvps.begin(), kvps.end(), [&] (std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
      return this->KeyCmpLess(a.first, b.first);
    });

    // Determine keys and values to stay in current leaf and copy others to new leaf
    uint16_t left_size = kvps.size() / 2;
    uint16_t i;
    for (i = 0; i < left_size; i++) {
      new_leaf_left->keys_[i] = kvps[i].first;
      new_leaf_left->values_[i] = kvps[i].second;
    }
    for (i = left_size; i < leaf->size_; i++) {
      new_leaf_right->keys_[i - left_size] = kvps[i].first;
      new_leaf_right->values_[i - left_size] = kvps[i].second;
    }

    // Determine pointer to new leaf to be pushed up to parent node
    KeyType new_key = new_leaf_right->keys_[0];

    // Update sizes of current and newly created leaf
    new_leaf_right->size_ = leaf->size_ - left_size;
    new_leaf_left->size_ = left_size;

    // Update neighbor pointers for leaf nodes
    new_leaf_left->left_ = leaf->left_.load();
    new_leaf_left->right_ = new_leaf_right;
    new_leaf_right->left_ = new_leaf_left;
    new_leaf_right->right_ = leaf->right_.load();

    if (LIKELY(left != nullptr)) {
      left->right_ = new_leaf_left;
      left->ReleaseWriteLatch();
    }
    leaf->MarkDeleted();
    leaf->ReleaseWriteLatch();
    if (LIKELY(right != nullptr)) {
      right->left_ = new_leaf_right;
      right->ReleaseWriteLatch();
    }

    // If split is on root (leaf)
    if (UNLIKELY(locked_nodes.empty())) {
      TERRIER_ASSERT(leaf == root_ && holds_tree_latch,
                     "somehow we had to split a leaf without having to modify the parent");
      // Create new root and update attributes for new root and tree
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = new_leaf_left;
      new_root->children_[1] = new_leaf_right;
      new_root->size_ = 1;
      BaseNode *old_root = root_;
      root_ = new_root;
      old_root->MarkDeleted();
      old_root->ReleaseWriteLatch();
      UnlatchRoot();
      return true;
    }

    InnerNode *old_node = nullptr, *new_node_left = nullptr, *new_node_right = nullptr;
    auto new_child_left = static_cast<BaseNode *>(new_leaf_left);
    auto new_child_right = static_cast<BaseNode *>(new_leaf_right);

    while (!locked_nodes.empty()) {
      old_node = locked_nodes.back();
      locked_nodes.pop_back();
      if (old_node->size_ < old_node->limit_) {
        break;
      }

      new_node_left = new InnerNode(this);
      new_node_right = new InnerNode(this);
      for (i = 0; i < old_node->size_; i++) {
        new_node_left->keys_[i] = old_node->keys_[i];
        new_node_left->children_[i] = old_node->children_[i].load();
      }
      new_node_left->children_[0] = old_node->children_[0].load();
      uint16_t child_index = traversal_indices.back();
      traversal_indices.pop_back();
      new_node_left->children_[child_index] = new_child_left;
      new_node_left->size_ = old_node->size_.load();

      for (uint16_t node_index = child_index; node_index < new_node_left->size_; node_index++) {
        std::swap(new_key, new_node_left->keys_[node_index]);
        BaseNode *temp = new_child_right;
        new_child_right = old_node->children_[node_index + 1].load();
        new_node_left->children_[node_index + 1] = temp;
      }

      uint16_t lower_length = new_node_left->size_ / 2;
      new_node_right->size_ = new_node_left->size_ - lower_length;
      new_node_right->keys_[new_node_right->size_ - 1] = new_key;
      new_node_right->children_[new_node_right->size_] = new_child_right;

      for (uint16_t node_index = lower_length + 1; node_index < new_node_left->size_; node_index++) {
        new_node_right->keys_[node_index - lower_length - 1] = new_node_left->keys_[node_index];
        new_node_right->children_[node_index - lower_length] = new_node_left->children_[node_index + 1].load();
      }
      new_node_right->children_[0] = new_node_left->children_[lower_length + 1].load();
      new_node_left->size_ = lower_length;
      new_key = new_node_left->keys_[lower_length];
      new_child_left = static_cast<BaseNode *>(new_node_left);
      new_child_right = static_cast<BaseNode *>(new_node_right);

      old_node->MarkDeleted();
      old_node->ReleaseWriteLatch();
    }


    if (old_node->size_ < old_node->limit_) {
      TERRIER_ASSERT(old_node->write_latch, "must hold write latch if not full");
      InnerNode* new_node = new InnerNode(this);
      new_node->size_ = old_node->size_.load();
      for (i = 0; i < old_node->size_; i++) {
        new_node->keys_[i] = old_node->keys_[i];
        new_node->children_[i + 1] = old_node->children_[i + 1].load();
      }
      new_node->children_[0] = old_node->children_[0].load();

      uint16_t child_index = traversal_indices.back();
      traversal_indices.pop_back();
      TERRIER_ASSERT(traversal_indices.size() == locked_nodes.size(),
          "they should be pushed onto and popped from equally");
      new_node->children_[child_index] = new_child_left;
      new_node->InsertInner(new_key, new_child_right);

      old_node->MarkDeleted();
      old_node->ReleaseWriteLatch();

      if (old_node == root_) {
        TERRIER_ASSERT(holds_tree_latch && locked_nodes.empty(),
            "must hold tree latch if held latch on root");
        root_ = new_node;
        UnlatchRoot();
      } else {
        TERRIER_ASSERT(!holds_tree_latch || !locked_nodes.empty(),
                       "if we are not at the root then we should not hold the tree latch"
                       "and should have released some write latches");
        child_index = traversal_indices.back();
        traversal_indices.pop_back();
        old_node = locked_nodes.back();
        locked_nodes.pop_back();
        old_node->children_[child_index] = new_node;
        old_node->ReleaseWriteLatch();
      }
    } else {
      TERRIER_ASSERT(locked_nodes.empty() && old_node == root_ && holds_tree_latch,
          "we should have split all the way to the top");
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = new_child_left;
      new_root->children_[1] = new_child_right;
      new_root->size_ = 1;
      root_ = new_root;
      old_node->MarkDeleted();
      old_node->ReleaseWriteLatch();
      UnlatchRoot();
    }

    return true;
  }

  void ScanKey(KeyType key, std::vector<ValueType> *values) {
    auto leaf = FindMinLeaf(key);
    while (true) {
      if (!leaf->ScanRange(key, &key, values)) {
        break;
      }
      leaf = leaf->right_;
      if (leaf == nullptr) {
        break;
      }
    }
  }

  bool Remove(KeyType key, ValueType value) {
    bool done = false;
    for (LeafNode* leaf = FindMinLeaf(key); leaf != nullptr && !done; leaf = leaf->right_) {
      BaseNode::ScopedWriteLatch (leaf);
      for (uint16_t i = 0; i < leaf->size_; i++) {
        if (leaf->IsReadable(i)) {
          if (KeyCmpEqual(key, leaf->keys_[i]) && value_eq_obj_(value, leaf->values_[i])) {
            leaf->MarkTombStone(i);
            return true;
          } else if (KeyCmpLess(key, leaf->keys_[i])) {
            done = true;
          }
        }
      }
    }
    return false;
  }

  void ScanDescending(KeyType lo, KeyType hi, std::vector<ValueType> *values) {
    for (LeafNode *l = FindMaxLeaf(hi); l != nullptr && !l->ScanRange(lo, &hi, values); l = l->left_) {}
  }

  void ScanDescendingLimit(KeyType lo, KeyType hi, uint32_t limit, std::vector<ValueType> *values) {
    for (LeafNode *l = FindMaxLeaf(hi); l != nullptr && values->size() < limit && !l->ScanRange(lo, &hi, values); l = l->left_) {}
    while (values->size() > limit) values->pop_back();
  }

  LeafNode* FindMinLeaf() {
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      n = inner_n->children_[0];
    }
    return static_cast<LeafNode *>(n);
  }

  LeafNode* FindMinLeaf(KeyType key) {
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      uint16_t child_index = inner_n->FindMinChild(key);
      n = inner_n->children_[child_index];
    }
    return static_cast<LeafNode *>(n);
  }

  LeafNode* FindMaxLeaf() {
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      n = inner_n->children_[inner_n->size_];
    }
    return static_cast<LeafNode *>(n);
  }

  LeafNode* FindMaxLeaf(KeyType key) {
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      InnerNode *inner_n = static_cast<InnerNode *>(n);
      n = inner_n->FindMaxChild(key);
    }
    return static_cast<LeafNode *>(n);
  }

  void LatchRoot() {
    bool t = true;
    bool f = false;
    while (!root_latch_.compare_exchange_strong(f, t)) {}
  }

  void UnlatchRoot() { root_latch_ = false; }

  std::atomic<uint64_t> epoch_;
  std::atomic<bool> root_latch_;
  std::atomic<BaseNode *> root_;
  std::atomic<uint64_t> structure_size_;
};

}  // namespace terrier::storage::index
