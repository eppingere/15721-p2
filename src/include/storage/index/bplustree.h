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
        epoch_(1),
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
  static const uint16_t INNER_NODE_OPTIMAL_FILL = BRANCH_FACTOR / 2;
  static const uint16_t LEAF_SIZE = 20;  // cannot ever be less than 3
  static const uint16_t LEAF_OPTIMAL_FILL = LEAF_SIZE / 2;
  static const uint64_t ALLOCATOR_START_SIZE = 10;
  static const uint64_t MAX_NUM_ACTIVE_EPOCHS = static_cast<uint64_t>(0x1) << 3;
  static const uint64_t WRITE_LOCK_MASK = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t NODE_TYPE_MASK = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t IS_DELETED_MASK = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t DELETE_EPOCH_MASK = ~(static_cast<uint64_t>(0xE) << 60);

  // Node types
  enum class NodeType : bool {
    LEAF = true,
    INNER_NODE = false,
  };

  enum class OptimisticResult : uint8_t {
    Success = 0,
    Failure = 1,
    RetryableFailure = 2,
  };

  class BaseNode {
   public:
    BaseNode() {}

    BaseNode(BPlusTree *tree, BPlusTree::NodeType t)
        : tree_(tree),
          info_(t == NodeType::LEAF ? NODE_TYPE_MASK : 0),
          size_(0),
          limit_(t == NodeType::LEAF ? LEAF_SIZE : BRANCH_FACTOR - 1) {}
    ~BaseNode() = default;

    BPlusTree *tree_;
//    std::atomic<bool> write_latch = false;
    common::SpinLatch write_latch;
    std::atomic<bool> deleted_ = false;
    std::atomic<uint64_t> deleted_epoch_ = 0;
    std::atomic<uint64_t> info_;
    std::atomic<uint16_t> size_, limit_;

    NodeType GetType() { return static_cast<NodeType>((info_ & NODE_TYPE_MASK) >> 62); }

    void MarkDeleted() {
      deleted_ = true;
      deleted_epoch_ = tree_->epoch_.load();
    }

    //
    void Reallocate() {
      write_latch = false;
      deleted_ = false;
      info_ = GetType() == NodeType::LEAF ? NODE_TYPE_MASK : 0;
      deleted_epoch_ = 0;
      size_ = 0;
    }

    void GetWriteLatch() {
//      bool t = true;
//      bool f = false;
//      while (!write_latch.compare_exchange_strong(f, t)) {}
//      TERRIER_ASSERT(write_latch, "should hold latch when latched");
      write_latch.Lock();
    }
    void ReleaseWriteLatch() {
//      TERRIER_ASSERT(write_latch, "should only be unlatching when latch is held");
//      write_latch = false;
//      TERRIER_ASSERT(!write_latch, "should actually release write latch");
      write_latch.Unlock();
    }

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
    InnerNode() = default;

    InnerNode(BPlusTree *tree) : BaseNode(tree, NodeType::INNER_NODE) {}
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
      typename BaseNode::ScopedWriteLatch l(this);
      for (uint16_t i = 0; i < this->size_; i++) {
        children_[i].load()->GetWriteLatch();
      }

      InnerNode *new_node;
      if (children_[0].load()->GetType() == NodeType::LEAF) {
        new_node = MergeAboveLeaves();
      } else {
        new_node = MergeAboveInnerNodes();
      }

      for (uint16_t i = 0; i < this->size_; i++) {
        children_[i].load()->MarkDeleted();
        children_[i].load()->ReleaseWriteLatch();
      }

      this->MarkDeleted();
      return new_node;
    }

    InnerNode *MergeAboveInnerNodes() {
      std::vector<std::pair<KeyType, BaseNode*>> kvps;

      InnerNode *child = static_cast<InnerNode *>(children_[0]);
      for (uint16_t j = 0; j < child->size_; j++) {
        kvps.emplace_back(std::pair<KeyType, BaseNode*>(child->keys_[j], child->children_[j + 1]));
      }

      for (uint16_t i = 1; i <= this->size_; i++) {
        child = static_cast<InnerNode *>(children_[i]);
        kvps.emplace_back(std::pair<KeyType, BaseNode*>(keys_[i - 1], child->children_[0]));
        for (uint16_t j = 0; j < child->size_; j++) {
          kvps.emplace_back(std::pair<KeyType, BaseNode*>(child->keys_[j], child->children_[j + 1]));
        }
      }

      // we put into each new child the min of (1) ceiling(total pairs + 1 (for additional pointer) / branch factor),
      // (2) BRANCH_FACTOR / 2
      uint64_t optimal_inner_node_size = (kvps.size() + static_cast<uint64_t>(BRANCH_FACTOR) - 1 + 1) /
                                   static_cast<uint64_t>(BRANCH_FACTOR);
      if (optimal_inner_node_size < INNER_NODE_OPTIMAL_FILL) {
        optimal_inner_node_size = INNER_NODE_OPTIMAL_FILL;
      }
      TERRIER_ASSERT(optimal_inner_node_size <= BRANCH_FACTOR,
                     "we should never have more than LEAF_SIZE many pairs in a leaf");
      TERRIER_ASSERT(optimal_inner_node_size * BRANCH_FACTOR >= kvps.size() + 1,
                     "we should have at least enough slots to cover all our tuples");

      InnerNode *new_node = new InnerNode(this->tree_);
      InnerNode *last_child = static_cast<InnerNode *>(children_[0].load())->children_[0];

      uint64_t new_node_index = 0;
      uint64_t kvps_index = 0;
      while (kvps_index < kvps.size()) {
        InnerNode *new_child = new InnerNode(this->tree_);
        new_child->children_[0] = last_child;

        uint64_t i;
        for (i = 0; i < optimal_inner_node_size && kvps_index + i < kvps.size(); i++) {
          new_child->keys_[i] = kvps[kvps_index + i].first;
          new_child->children_[i + 1] = kvps[kvps_index + i].second;
        }

        new_child->size_ = i;
        if (kvps_index + i < kvps.size()) {
          new_node->keys_[new_node_index] = kvps[kvps_index + i];
          last_child = kvps[kvps_index + i];
        }

        kvps_index += i;
        new_node_index++;
      }

      new_node->size_ = new_node_index;
      return new_node;
    }

    InnerNode *MergeAboveLeaves() {
      std::vector<std::pair<KeyType, ValueType>> kvps;
      for (uint16_t i = 0; i < this->size_; i++) {
        auto *leaf = static_cast<LeafNode *>(children_[i]);
        for (uint16_t j = 0; j < leaf->size_; j++) {
          if (leaf->IsReadable(j)) {
            kvps.emplace_back(std::pair<KeyType, ValueType>(leaf->keys_[j], leaf->values_[j]));
          }
        }
      }

      sort(kvps.begin(), kvps.end(), [&] (std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
        return this->tree_->KeyCmpLess(a.first, b.first);
      });

      // we put into each new leaf the min of (1) ceiling(total pairs / branch factor), (2) LEAF_SIZE / 2
      uint16_t optimal_leaf_size = (kvps.size() + static_cast<uint64_t>(BRANCH_FACTOR) - 1) /
                                   static_cast<uint64_t>(BRANCH_FACTOR);
      if (optimal_leaf_size < LEAF_OPTIMAL_FILL) {
        optimal_leaf_size = LEAF_OPTIMAL_FILL;
      }
      TERRIER_ASSERT(optimal_leaf_size <= LEAF_SIZE,
          "we should never have more than LEAF_SIZE many pairs in a leaf");
      TERRIER_ASSERT(optimal_leaf_size * BRANCH_FACTOR >= kvps.size(),
          "we should have at least enough slots to cover all our tuples");

      InnerNode *new_node = new InnerNode(this->tree_);
      LeafNode *new_leaf = new LeafNode(this->tree_);
      uint16_t i;
      for (i = 0; i < optimal_leaf_size && i < kvps.size(); i++) {
        new_leaf->keys_[i] = kvps[i].first;
        new_leaf->values_[i] = kvps[i].second;
      }
      new_leaf->size_ = i;
      new_node->children_[0] = new_node;

      uint64_t allocated_index = 0;
      uint64_t inner_node_index = 0;
      while (allocated_index < kvps.size()) {
        TERRIER_ASSERT(inner_node_index < BRANCH_FACTOR, "must have at most branch factor many children");
        new_leaf = new LeafNode(this->tree_);
        for (i = 0;
             i < optimal_leaf_size && allocated_index + static_cast<uint64_t>(i) < kvps.size();
             i++) {
          new_leaf->keys_[i] = kvps[i + allocated_index].first;
          new_leaf->values_[i] = kvps[i + allocated_index].second;
        }
        new_leaf->size_ = i;
        new_node->children_[inner_node_index + 1] = new_node;
        new_node->keys_[inner_node_index] = new_leaf->keys_[0];

        new_leaf->left_ = new_node->children_[inner_node_index];
        new_node->children_[inner_node_index]->right_ = new_leaf;
        new_leaf->left_ = new_node->children_[inner_node_index];

        allocated_index += static_cast<uint64_t>(i);
        inner_node_index++;
      }
      new_node->size_ = inner_node_index;

      LeafNode* right = static_cast<LeafNode *>(children_[this->size_])->right;
      new_node->children_[inner_node_index]->right = right;
      if (LIKELY(right != nullptr)) {
        right->left_ = new_node->children_[inner_node_index];
      }

      return new_node;
    }

    std::atomic<BaseNode *> children_[BRANCH_FACTOR];
    KeyType keys_[BRANCH_FACTOR - 1];
  };

  class LeafNode : public BaseNode {
   public:
    LeafNode() = default;

    LeafNode(BPlusTree *tree) : BaseNode(tree, NodeType::LEAF), left_(nullptr), right_(nullptr) {}
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

      // TODO(deepayan): see if insert to the right works out instead of forcing split here
      if (this->size_ >= this->limit_) {
        return false;
      }

      keys_[this->size_] = key;
      values_[this->size_] = value;
      this->size_++;
      return true;
    }

    bool ScanRange(KeyType low, KeyType *hi, std::vector<ValueType> *values, std::function<bool(std::pair<KeyType, ValueType>)> predicate) {
      bool res = true;
      std::vector<std::pair<KeyType, ValueType>> temp_values;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (IsReadable(i)) {
          if (this->tree_->KeyCmpLessEqual(low, keys_[i]) &&
              (hi == nullptr || this->tree_->KeyCmpLessEqual(keys_[i], *hi))) {
            auto pair = std::pair<KeyType, ValueType>(keys_[i], values_[i]);
            if (predicate(pair)) temp_values.emplace_back(std::pair<KeyType, ValueType>(keys_[i], values_[i]));
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

    bool ScanRangeReverse(KeyType low, KeyType *hi, std::vector<ValueType> *values, std::function<bool(ValueType)> predicate) {
      bool res = true;
      std::vector<std::pair<KeyType, ValueType>> temp_values;
      for (uint16_t i = this->size_ - 1; i < this->size_; i--) {
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
        return this->tree_->KeyCmpGreater(a.first, b.first);
      });

      for (uint16_t i = 0; i < temp_values.size(); i++) {
        if (predicate(temp_values[i].second)) values->emplace_back(temp_values[i].second);
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

//  class InnerNodeAllocator {
//    static const uint64_t ALLOCATOR_ARRAY_SIZE = 64;
//
//    class InnerNodeAllocatorArray {
//     public:
//      InnerNodeAllocatorArray(BPlusTree<KeyType, ValueType> *tree) {
//        for (uint64_t i = 0; i < ALLOCATOR_ARRAY_SIZE; i++) {
//          new (&(array_[i])) InnerNode(tree);
//        }
//      }
//
//      bool Allocate(uint64_t i) {
//        while (true) {
//          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//          uint64_t new_value =
//              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
//          uint64_t available = old_value >> (i % BITS_IN_UINT64);
//          if (!static_cast<bool>(available & static_cast<uint64_t>(0x1))) return false;
//          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
//          return true;
//        }
//      }
//
//      bool IsAvailable(uint64_t i) {
//        uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//        return !static_cast<bool>((old_value >> (i % BITS_IN_UINT64)) & static_cast<uint64_t>(0x1));
//      }
//
//      void Reclaim(uint64_t i) {
//        while (true) {
//          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//          uint64_t new_value =
//              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
//          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
//          return;
//        }
//      }
//
//      std::atomic<uint64_t> allocated_masks_[(ALLOCATOR_ARRAY_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64] = {};
//      LeafNode array_[ALLOCATOR_ARRAY_SIZE];
//    };
//
//    void ReclaimOldNodes(uint64_t safe_to_delete_epoch) {
//      while (!resizing_.compare_exchange_strong(false, true)) {}
//      for (uint64_t i = 0; i < size_; i++) {
//        for (uint64_t j = 0; j < ALLOCATOR_ARRAY_SIZE; j++) {
//          if (!table_[i]->IsAvailable(j)) {
//            BaseNode node = static_cast<BaseNode>(table_[i]->array_[j]);
//            if (node.deleted_ && node.deleted_epoch_ <= safe_to_delete_epoch) {
//              table_[i]->Reclaim(j);
//            }
//          }
//        }
//      }
//    }
//
//   public:
//    InnerNodeAllocator(uint64_t starting_size, BPlusTree<KeyType, ValueType> *tree) :
//        tree_(tree),
//        resizing_(false),
//        size_((starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE),
//        table_(new InnerNodeAllocatorArray*[(starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE]) {
//      for (uint64_t i = 0; i < (starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE; i++) {
//        table_[i] = new InnerNodeAllocatorArray(tree_);
//      }
//    }
//
//    InnerNode* NewNode() {
//      do {
//        for (uint64_t i = 0; i < size_; i++) {
//          uint64_t j;
//          for (j = 0; j < ALLOCATOR_ARRAY_SIZE && !table_[i]->Allocate(j); j++) {}
//          if (j != ALLOCATOR_ARRAY_SIZE) {
//            InnerNode *node = table_[i]->array_ + j;
//            node->Reallocate();
//            return (table_[i]->array_ + j);
//          }
//        }
//      } while (resizing_);
//
//      while (!resizing_.compare_exchange_strong(false, true)) {}
//      InnerNodeAllocatorArray ** new_table = new InnerNodeAllocatorArray*[2 * size_];
//      for (uint64_t i = 0; i < size_; i++) {
//        new_table[i] = table_[i];
//      }
//
//      for (uint64_t i = size_; i < 2 * size_; i++) {
//        new_table[i] = new InnerNodeAllocatorArray();
//      }
//
//      InnerNodeAllocatorArray ** old_table = table_;
//      table_ = new_table;
//      resizing_ = false;
//      delete old_table;
//    }
//
//   private:
//    BPlusTree<KeyType, ValueType> *tree_;
//    std::atomic<bool> resizing_;
//    std::atomic<uint64_t> size_;
//    std::atomic<InnerNodeAllocatorArray **> table_;
//  };

//  class LeafNodeAllocator {
//    static const uint64_t ALLOCATOR_ARRAY_SIZE = 64;
//
//    class LeafNodeAllocatorArray {
//     public:
//      LeafNodeAllocatorArray(BPlusTree<KeyType, ValueType> tree) {
//        for (uint64_t i = 0; i < ALLOCATOR_ARRAY_SIZE; i++) {
//          new (&(array_[i])) LeafNode(tree);
//        }
//      }
//
//      bool Allocate(uint64_t i) {
//        while (true) {
//          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//          uint64_t new_value =
//              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
//          uint64_t available = old_value >> (i % BITS_IN_UINT64);
//          if (!static_cast<bool>(available & static_cast<uint64_t>(0x1))) return false;
//          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
//          return true;
//        }
//      }
//
//      bool IsAvailable(uint64_t i) {
//        uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//        return !static_cast<bool>((old_value >> (i % BITS_IN_UINT64)) & static_cast<uint64_t>(0x1));
//      }
//
//      void Reclaim(uint64_t i) {
//        while (true) {
//          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
//          uint64_t new_value =
//              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
//          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
//          return;
//        }
//      }
//
//      std::atomic<uint64_t> allocated_masks_[(ALLOCATOR_ARRAY_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64] = {};
//      LeafNode array_[];
//    };
//
//    void ReclaimOldNodes(uint64_t safe_to_delete_epoch) {
//      while (!resizing_.compare_exchange_strong(false, true)) {}
//      for (uint64_t i = 0; i < size_; i++) {
//        for (uint64_t j = 0; j < ALLOCATOR_ARRAY_SIZE; j++) {
//          if (!table_[i]->IsAvailable(j)) {
//            BaseNode node = static_cast<BaseNode>(table_[i]->array_[j]);
//            if (node.deleted_ && node.deleted_epoch_ <= safe_to_delete_epoch) {
//              table_[i]->Reclaim(j);
//            }
//          }
//        }
//      }
//    }
//
//   public:
//    LeafNodeAllocator(uint64_t starting_size, BPlusTree<KeyType, ValueType> *tree) :
//        tree_(tree),
//        resizing_(false),
//        size_((starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE),
//        table_(new LeafNodeAllocatorArray[(starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE]) {
//      for (uint64_t i = 0; i < (starting_size + ALLOCATOR_ARRAY_SIZE - 1) / ALLOCATOR_ARRAY_SIZE; i++) {
//        new (&table_[i]) LeafNodeAllocatorArray(tree);
//      }
//    }
//
//    LeafNode* NewNode() {
//      do {
//        for (uint64_t i = 0; i < size_; i++) {
//          uint64_t j;
//          for (j = 0; j < ALLOCATOR_ARRAY_SIZE && !table_[i]->Allocate(j); j++) {}
//          if (j != ALLOCATOR_ARRAY_SIZE) {
//            LeafNode *node = table_[i]->array_ + j;
//            node->Reallocate();
//            return (table_[i]->array_ + j);
//          }
//        }
//      } while (resizing_);
//
//      while (!resizing_.compare_exchange_strong(false, true)) {}
//      LeafNodeAllocatorArray ** new_table = new LeafNodeAllocatorArray*[2 * size_];
//      for (uint64_t i = 0; i < size_; i++) {
//        new_table[i] = table_[i];
//      }
//
//      for (uint64_t i = size_; i < 2 * size_; i++) {
//        new (&table_[i]) LeafNodeAllocatorArray(this->tree);
//      }
//
//      LeafNodeAllocatorArray ** old_table = table_;
//      table_ = new_table;
//      resizing_ = false;
//      delete old_table;
//    }
//
//   private:
//    BPlusTree<KeyType, ValueType> *tree_;
//    std::atomic<bool> resizing_;
//    std::atomic<uint64_t> size_;
//    std::atomic<LeafNodeAllocatorArray *> table_;
//  };

  void ReclaimOldNodes() {
    uint64_t old_epoch = epoch_;
    uint64_t iter = 1;
    while (active_epochs_[(old_epoch + iter) % MAX_NUM_ACTIVE_EPOCHS] != 0) {}
    for (iter = 2;
         iter < MAX_NUM_ACTIVE_EPOCHS - 1 && active_epochs_[(old_epoch + iter) % MAX_NUM_ACTIVE_EPOCHS] == 0;
         iter++) {}

//    uint64_t safe_iter = iter - 1;
//    uint64_t safe_to_delete_epoch = epoch_ + safe_iter < MAX_NUM_ACTIVE_EPOCHS ? 0 :
//                                    epoch_ + safe_iter - MAX_NUM_ACTIVE_EPOCHS;

    epoch_++;

//    inner_node_allocator_.ReclaimOldNodes(safe_to_delete_epoch);
//    leaf_node_allocator_.ReclaimOldNodes(safe_to_delete_epoch);
  }

  void RunGarbageCollection() {
    CompressTree();
    ReclaimOldNodes();
  }

  uint64_t StartFunction() {
    uint64_t current_epoch = epoch_;
    active_epochs_[current_epoch % MAX_NUM_ACTIVE_EPOCHS]++;
    return current_epoch;
  }

  void EndFunction(uint64_t start_epoch) {
    active_epochs_[start_epoch % MAX_NUM_ACTIVE_EPOCHS]--;
  }

  inline bool OptimisticInsert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    return FindMinLeaf(key)->InsertLeaf(key, val, predicate, predicate_satisfied);
  }

  bool InsertHelper(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    if (OptimisticInsert(predicate, key, val, predicate_satisfied)) {
      return true;
    }

    std::vector<InnerNode *> locked_nodes;
    std::vector<uint16_t> traversal_indices;

    LatchRoot();
    bool holds_tree_latch = true;
    BaseNode *n = this->root_;
    TERRIER_ASSERT(!n->deleted_, "we should never traverse deleted nodes");

    // Find minimum leaf that stores would store key, taking locks as we go down tree
    while (n->GetType() != NodeType::LEAF) {
      auto inner_n = static_cast<InnerNode *>(n);
      inner_n->GetWriteLatch();
      if (n->size_ < n->limit_) {
        if (holds_tree_latch && locked_nodes.size() > 1) {
          UnlatchRoot();
          holds_tree_latch = false;
        }
        for (uint64_t i = 0; (!locked_nodes.empty()) && i < locked_nodes.size() - 1; i++) {
          locked_nodes[i]->ReleaseWriteLatch();
        }
        if (!locked_nodes.empty()) {
          InnerNode * last = locked_nodes.back();
          uint16_t last_index = traversal_indices.back();
          locked_nodes.clear();
          traversal_indices.clear();
          locked_nodes.emplace_back(last);
          traversal_indices.emplace_back(last_index);
        }

      }
      uint16_t child_index = inner_n->FindMinChild(key);
      n = inner_n->children_[child_index];
      locked_nodes.emplace_back(inner_n);
      traversal_indices.emplace_back(child_index);
      TERRIER_ASSERT(!n->deleted_, "we should never traverse deleted nodes");
    }
    TERRIER_ASSERT(!n->deleted_, "we should never traverse deleted nodes");

//    for (InnerNode* node : locked_nodes)
//      TERRIER_ASSERT(node->write_latch, "locked nodes should actually be locked");


    // If leaf is not full, insert, unlock all parent nodes and return
    auto leaf = static_cast<LeafNode *>(n);
    if (leaf->InsertLeaf(key, val, predicate, predicate_satisfied) || *predicate_satisfied) {
      if (holds_tree_latch) {
        UnlatchRoot();
        holds_tree_latch = false;
      }
      for (InnerNode* node : locked_nodes)
        node->ReleaseWriteLatch();

      return !(*predicate_satisfied);
    }

    // returned false because of no tomb stones;

    LeafNode *left, *right;
    while (true) {
      left = leaf->left_.load();
      right = leaf->right_.load();
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
          if (LIKELY(left != nullptr)) {
            left->ReleaseWriteLatch();
          }
          leaf->ReleaseWriteLatch();
          right->ReleaseWriteLatch();
          continue;
        }
      }
      break;
    }

    TERRIER_ASSERT(!leaf->deleted_ && (left == nullptr || !left->deleted_) && (right == nullptr || !right->deleted_), "none of leaf right and left should be marked as deleted");


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

    // TODO(deepayan): check if this actually is hit and sorts
    // Determine keys and values to stay in current leaf and copy others to new leaf
    uint16_t left_size = kvps.size() / 2;
    uint16_t i;
    for (i = 0; i < left_size; i++) {
      new_leaf_left->keys_[i] = kvps[i].first;
      new_leaf_left->values_[i] = kvps[i].second;
    }
    for (i = left_size; i < kvps.size(); i++) {
      new_leaf_right->keys_[i - left_size] = kvps[i].first;
      new_leaf_right->values_[i - left_size] = kvps[i].second;
    }

    // Determine pointer to new leaf to be pushed up to parent node
    KeyType new_key = new_leaf_right->keys_[0];

    // Update sizes of current and newly created leaf
    new_leaf_right->size_ = kvps.size() - left_size;
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
      TERRIER_ASSERT(leaf == root_, "we had to split a leaf without having to modify the parent");
      TERRIER_ASSERT(holds_tree_latch,
                     "we are trying to split the root without a latch on the tree");
      // Create new root and update attributes for new root and tree
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = new_leaf_left;
      new_root->children_[1] = new_leaf_right;
      new_root->size_ = 1;
      BaseNode *old_root = root_;
      root_ = new_root;
      old_root->MarkDeleted();
      UnlatchRoot();
      holds_tree_latch = false;
      return true;
    }

    InnerNode *old_node = nullptr, *new_node_left = nullptr, *new_node_right = nullptr;
    auto new_child_left = static_cast<BaseNode *>(new_leaf_left);
    auto new_child_right = static_cast<BaseNode *>(new_leaf_right);

    uint64_t num_popped = 0;
    while (!locked_nodes.empty()) {
      old_node = locked_nodes.back();
      locked_nodes.pop_back();
      num_popped++;
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

    int which_case = 0;
    if (old_node->size_ < old_node->limit_) {
//      TERRIER_ASSERT(old_node->write_latch, "must hold write latch if not full");
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
        which_case = 1;
        TERRIER_ASSERT(holds_tree_latch && locked_nodes.empty(),
            "must hold tree latch if held latch on root");
        root_ = new_node;
        UnlatchRoot();
        holds_tree_latch = false;
      } else {
        which_case = 2;
        TERRIER_ASSERT(!holds_tree_latch || !locked_nodes.empty(),
                       "if we are not at the root then we should not hold the tree latch"
                       "and should have released some write latches");
        child_index = traversal_indices.back();
        traversal_indices.pop_back();
        old_node = locked_nodes.back();
        locked_nodes.pop_back();
        num_popped++;
        old_node->children_[child_index] = new_node;
        old_node->ReleaseWriteLatch();
        if (holds_tree_latch) {
          TERRIER_ASSERT(old_node == root_, "if we hold the tree latch, then we are adjusting the root");
          UnlatchRoot();
          holds_tree_latch = false;
        }
      }
    } else {
      which_case = 3;
      TERRIER_ASSERT(locked_nodes.empty() && old_node == root_ && holds_tree_latch,
          "we should have split all the way to the top");
      auto new_root = new InnerNode(this);
      new_root->keys_[0] = new_key;
      new_root->children_[0] = new_child_left;
      new_root->children_[1] = new_child_right;
      new_root->size_ = 1;
      root_ = new_root;
      UnlatchRoot();
      holds_tree_latch = false;
    }

    TERRIER_ASSERT(locked_nodes.size() == 0, "we should have unlocked all locked nodes");
    TERRIER_ASSERT(!holds_tree_latch, "should not hold the rootlatch on return");
    return true;
  }

  bool Insert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    uint64_t epoch = StartFunction();
    bool result = InsertHelper(predicate, key, val, predicate_satisfied);
    EndFunction(epoch);
    return result;
  }

  void ScanKeyHelper(KeyType key, std::vector<ValueType> *values, std::function<bool(ValueType)> predicate) {
    bool done = false;
    for (auto* leaf = FindMinLeaf(key); leaf != nullptr && !done; leaf = leaf->right_) {
      for (uint16_t i = 0; i < leaf->size_; i++) {
        if (leaf->IsReadable(i)) {
          if (KeyCmpEqual(key, leaf->keys_[i]) && predicate(leaf->values_[i])) {
            values->emplace_back(leaf->values_[i]);
          } else if (KeyCmpLess(key, leaf->keys_[i])) {
            done = true;
          }
        }
      }
    }

  }

  void ScanKey(KeyType key, std::vector<ValueType> *values, std::function<bool(ValueType)> predicate) {
    uint64_t epoch = StartFunction();
    ScanKeyHelper(key, values, predicate);
    EndFunction(epoch);
  }

  bool RemoveHelper(KeyType key, ValueType value) {
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

  bool Remove(KeyType key, ValueType value) {
    uint64_t epoch = StartFunction();
    bool result = RemoveHelper(key, value);
    EndFunction(epoch);
    return result;
  }

  void MergeToDepth(InnerNode *node, uint64_t max_depth, uint64_t current_depth) {
    if (UNLIKELY(node->GetType() == NodeType::LEAF || node->deleted_)) {
      return;
    }

    if (max_depth <= current_depth) {
      typename BaseNode::ScopedWriteLatch l(node);
      if (node->deleted_) {
        return;
      }
      for (uint64_t i = 0; i <= node->size_; i++) {
        if (UNLIKELY(node->children_[i].load()->GetType() == NodeType::LEAF)) continue;
        InnerNode* child = static_cast<InnerNode *>(node->children_[i]);
        node->children_[i] = child->Merge();
      }
      return;
    }

    for (uint16_t i = 0; i <= node->size_; i++) {
      MergeToDepth(node->children_[i], max_depth, current_depth + 1);
    }
  }

  void CompressTree() {
    uint64_t depth = GetDepth();
    if (depth == 1) {
      return;
    }
    if (depth == 2) {
      LatchRoot();
      if (LIKELY(root_.load()->GetType() == NodeType::INNER_NODE)) {
        root_ = static_cast<InnerNode *>(root_.load())->Merge();
      }
      UnlatchRoot();
      return;
    }
    for (uint64_t d = depth - 2; d > 1; d--) {
      MergeToDepth(root_.load(), d, 1);
    }
    LatchRoot();
    if (LIKELY(root_.load()->GetType() == NodeType::INNER_NODE)) {
      MergeToDepth(root_.load(), 1, 1);
      root_ = static_cast<InnerNode *>(root_.load())->Merge();
    }
    UnlatchRoot();
  }

  uint64_t GetDepth() {
    uint64_t d = 1;
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      n = static_cast<InnerNode *>(n)->children_[0];
      d++;
    }
    return d;
  }

  LeafNode* FindMinLeaf() {
    while (true) {
      OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        InnerNode *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->children_[0];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  LeafNode* FindMinLeaf(KeyType key) {
    while (true) {
    OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        InnerNode *inner_n = static_cast<InnerNode *>(n);
        uint16_t child_index = inner_n->FindMinChild(key);
        n = inner_n->children_[child_index];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  LeafNode* FindMaxLeaf() {
    while (true) {
      OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        InnerNode *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->children_[n->size_];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  LeafNode* FindMaxLeaf(KeyType key) {
    while (true) {
      OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        InnerNode *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->FindMaxChild(key);
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  void LatchRoot() {
//    bool t = true;
//    bool f = false;
//    while (!root_latch_.compare_exchange_strong(f, t)) {}
//    TERRIER_ASSERT(root_latch_, "root latch should be held");
    root_latch_.Lock();
  }

  void UnlatchRoot() {
//    TERRIER_ASSERT(root_latch_, "root latch should be held");
//    root_latch_ = false;
    root_latch_.Unlock();
  }

  std::atomic<uint64_t> active_epochs_[MAX_NUM_ACTIVE_EPOCHS] = {};
//  InnerNodeAllocator inner_node_allocator_ = InnerNodeAllocator(ALLOCATOR_START_SIZE, this);
//  LeafNodeAllocator leaf_node_allocator_ = LeafNodeAllocator(ALLOCATOR_START_SIZE, this);
  std::atomic<uint64_t> epoch_ = 1;
  common::SpinLatch root_latch_;
//  std::atomic<bool> root_latch_ = false;
  std::atomic<BaseNode *> root_;
  std::atomic<uint64_t> structure_size_;
};

}  // namespace terrier::storage::index
