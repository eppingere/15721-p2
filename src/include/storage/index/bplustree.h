#pragma once

#include <common/shared_latch.h>
#include <execution/util/execution_common.h>
#include <storage/storage_defs.h>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "tbb/concurrent_unordered_set.h"

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
   * Constructor - Set up initial environment for BPlusTree
   *
   * Any tree instance will start with an empty leaf node as the root, which will then be filled.
   *
   * Some properties of the tree should be specified in the argument.
   */
  explicit BPlusTree(KeyComparator p_key_cmp_obj = KeyComparator{},
                     KeyEqualityChecker p_key_eq_obj = KeyEqualityChecker{}, KeyHashFunc p_key_hash_obj = KeyHashFunc{},
                     ValueEqualityChecker p_value_eq_obj = ValueEqualityChecker{})
      : key_cmp_obj_{p_key_cmp_obj}, key_eq_obj_{p_key_eq_obj}, value_eq_obj_{p_value_eq_obj}, epoch_(1) {
    root_ = static_cast<BaseNode *>(new LeafNode(this));
  }

  ~BPlusTree() = default;

  // Tunable parameters
  static const uint16_t BRANCH_FACTOR = 20;
  static const uint16_t INNER_NODE_OPTIMAL_FILL = BRANCH_FACTOR / 2;
  static const uint16_t LEAF_SIZE = 128;  // cannot ever be less than 3
  static const uint16_t LEAF_OPTIMAL_FILL = LEAF_SIZE / 2;
  static const uint64_t MAX_NUM_ACTIVE_EPOCHS = static_cast<uint64_t>(0x1) << 3;
  static const uint64_t ALLOCATOR_ARRAY_SIZE = 64;

  // Constants
  static const uint64_t BITS_IN_UINT64 = 8 * sizeof(uint64_t);

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

  /*
   * NodeType : an alias for our two node types
   */
  enum class NodeType : bool {
    LEAF = true,
    INNER_NODE = false,
  };

  /*
   * OptimisticResult: is a return value for the different results than an optimistic function can take on
   */
  enum class OptimisticResult : uint8_t {
    Success = 0,
    Failure = 1,
    RetryableFailure = 2,
  };

  class BPlusTreeLatch {
   public:
    void Lock() { latch_.Lock(); }
    void Unlock() { latch_.Unlock(); }

   private:
    common::SpinLatch latch_;
  };

  /*
   * BaseNode is a super class for our two node types. It contains all their shared metadata as well as functions that
   * are shared among them. All nodes do not require latches when being read. Only require latches when being written
   * to.
   */
  class BaseNode {
   public:
    BaseNode() = default;

    BaseNode(BPlusTree *tree, BPlusTree::NodeType t) : tree_(tree), size_(0), type_(t) {}
    ~BaseNode() = default;

    BPlusTree *tree_;
    std::atomic<uint64_t> deleted_epoch_;
    std::atomic<uint16_t> size_;
    common::SpinLatch write_latch_;
    std::atomic<bool> deleted_ = false;
    NodeType type_;

    NodeType GetType() { return type_; }

    void MarkDeleted() {
      deleted_ = true;
      deleted_epoch_ = tree_->epoch_.load();
    }

    uint16_t GetLimit() { return GetType() == NodeType::LEAF ? LEAF_SIZE : BRANCH_FACTOR - 1; }

    void Lock() {
      //      bool t = true;
      //      bool f = false;
      //      while (!write_latch_.compare_exchange_strong(f, t)) {}
      //      TERRIER_ASSERT(write_latch_, "should hold latch when latched");
      write_latch_.Lock();
    }
    void Unlock() {
      //      TERRIER_ASSERT(write_latch_, "should only be unlatching when latch is held");
      //      write_latch_ = false;
      //      TERRIER_ASSERT(!write_latch_, "should actually release write latch");
      write_latch_.Unlock();
    }

    class ScopedWriteLatch {
     public:
      explicit ScopedWriteLatch(BaseNode *node) : node_(node) { node_->Lock(); }
      ~ScopedWriteLatch() { node_->Unlock(); }
      BaseNode *node_;
    };
  };

  /*
   * InnerNode: is a subclass of BaseNode. It serves as as BRANCH_FACTOR-way inner node of the tree. It contains
   * (BRANCH_FACTOR - 1) many keys that partition the ranges that inhabit its children. In inner node size tracks the
   * number of keys in the tree, which is on less than the number of children. Each key k at index i partitions children
   * at index i to contain keys less than k (if there are duplicate keys that span across the children at i and i+1 then
   * the ith child can contain keys less or equal to k) and index i+1 to contain keys greater or equal to k.
   */
  class InnerNode : public BaseNode {
   public:
    InnerNode() = default;

    explicit InnerNode(BPlusTree *tree) : BaseNode(tree, NodeType::INNER_NODE) {}
    ~InnerNode() = default;

    /// FindMinChild: Finds the left-most child of this that could contain the given key
    /// \param key : the given key
    /// \return an index into children_ of the left-most child of this node that could contain the given key
    uint16_t FindMinChild(KeyType key) {
      uint16_t i;
      for (i = 0; i < this->size_.load(); i++)
        if (this->tree_->KeyCmpLessEqual(key, this->keys_[i])) return i;

      return i;
    }

    /// FindMaxChild: Finds the right-most child of this that could contain the given key
    /// \param key : the given key
    /// \return an index into children_ of the right-most child of this node that could contain the given key
    BaseNode *FindMaxChild(KeyType key) {
      for (uint16_t i = this->size_ - 1; i < this->size_; i--)
        if (this->tree_->KeyCmpGreaterEqual(key, keys_[i])) return children_[i + 1];

      return children_[0];
    }

    /// InsertInner adds to the node this new key and its right child. Can only be called on not visible nodes.
    /// \param key
    /// \param child the right child of the given key
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

    /// Merge generates a pointer to a merged version of this. Requires lock to be held on parent or the root if root
    /// \return a pointer to a new node that balances this node and its children. Marks this node and all children
    /// as deleted
    InnerNode *Merge() {
      typename BaseNode::ScopedWriteLatch l(this);
      TERRIER_ASSERT(!this->deleted_, "we should never merge a deleted node");

      bool left_locked = false;
      bool right_locked = false;
      LeafNode *left = nullptr, *right = nullptr;
      while (true) {
        if (children_[0].load()->GetType() == NodeType::LEAF &&
            static_cast<LeafNode *>(children_[0].load())->left_ != nullptr) {
          left = static_cast<LeafNode *>(children_[0].load())->left_;
          left->Lock();
          if (left->deleted_ || left != static_cast<LeafNode *>(children_[0].load())->left_) {
            left->Unlock();
          }
          left_locked = true;
        }
        break;
      }

      for (uint16_t i = 0; i < this->size_; i++) {
        children_[i].load()->Lock();
      }

      while (true) {
        if (children_[this->size_].load()->GetType() == NodeType::LEAF &&
            static_cast<LeafNode *>(children_[this->size_].load())->right_ != nullptr) {
          right = static_cast<LeafNode *>(children_[this->size_].load())->right_;
          right->Lock();
          if (right->deleted_ || right != static_cast<LeafNode *>(children_[this->size_].load())->right_) {
            right->Unlock();
          }
          right_locked = true;
        }
        break;
      }

      InnerNode *new_node;
      if (children_[0].load()->GetType() == NodeType::LEAF) {
        new_node = MergeAboveLeaves();
      } else {
        new_node = MergeAboveInnerNodes();
      }

      if (left_locked) {
        left->Unlock();
      }
      for (uint16_t i = 0; i < this->size_; i++) {
        children_[i].load()->MarkDeleted();
        children_[i].load()->Unlock();
      }
      if (right_locked) {
        right->Unlock();
      }

      this->MarkDeleted();
      return new_node;
    }

    /// MergeAboveInnerNodes helper function for Merge. Implements merge if the innernode's children are innernodes
    /// \return pointer to merged node
    InnerNode *MergeAboveInnerNodes() {
      std::vector<std::pair<KeyType, BaseNode *>> kvps;

      auto *child = static_cast<InnerNode *>(children_[0].load());
      for (uint16_t j = 0; j < child->size_; j++) {
        kvps.emplace_back(std::pair<KeyType, BaseNode *>(child->keys_[j], child->children_[j + 1].load()));
      }

      for (uint16_t i = 1; i <= this->size_; i++) {
        child = static_cast<InnerNode *>(children_[i].load());
        kvps.emplace_back(std::pair<KeyType, BaseNode *>(keys_[i - 1], child->children_[0].load()));
        for (uint16_t j = 0; j < child->size_; j++) {
          kvps.emplace_back(std::pair<KeyType, BaseNode *>(child->keys_[j], child->children_[j + 1].load()));
        }
      }

      // we put into each new child the min of (1) ceiling(total pairs + 1 (for additional pointer) / branch factor),
      // (2) BRANCH_FACTOR / 2
      uint64_t optimal_inner_node_size =
          (kvps.size() + static_cast<uint64_t>(BRANCH_FACTOR) - 1 + 1) / static_cast<uint64_t>(BRANCH_FACTOR);
      if (optimal_inner_node_size < INNER_NODE_OPTIMAL_FILL) {
        optimal_inner_node_size = INNER_NODE_OPTIMAL_FILL;
      }
      TERRIER_ASSERT(optimal_inner_node_size <= BRANCH_FACTOR,
                     "we should never have more than LEAF_SIZE many pairs in a leaf");
      TERRIER_ASSERT(optimal_inner_node_size * BRANCH_FACTOR >= kvps.size() + 1,
                     "we should have at least enough slots to cover all our tuples");

      auto *new_node = new InnerNode(this->tree_);
      auto *last_child = static_cast<InnerNode *>(static_cast<InnerNode *>(children_[0].load())->children_[0].load());

      uint64_t new_node_index = 0;
      uint64_t kvps_index = 0;
      do {
        auto *new_child = new InnerNode(this->tree_);
        new_child->children_[0] = last_child;

        uint64_t i;
        for (i = 0; i < optimal_inner_node_size && kvps_index + i < kvps.size(); i++) {
          new_child->keys_[i] = kvps[kvps_index + i].first;
          new_child->children_[i + 1] = kvps[kvps_index + i].second;
        }

        new_child->size_ = i;
        if (kvps_index + i < kvps.size()) {
          new_node->keys_[new_node_index] = kvps[kvps_index + i].first;
          last_child = static_cast<InnerNode *>(kvps[kvps_index + i].second);
        }

        new_node->children_[new_node_index] = new_child;

        kvps_index += i;
        new_node_index++;
      } while (kvps_index < kvps.size());

      new_node->size_ = new_node_index - 1;
      return new_node;
    }

    /// MergeAboveLeaves helper function for Merge. Implements merge if the innernode's children are leaves
    /// \return pointer to merged node
    InnerNode *MergeAboveLeaves() {
      std::vector<std::pair<KeyType, ValueType>> kvps;
      for (uint16_t i = 0; i <= this->size_; i++) {
        auto *leaf = static_cast<LeafNode *>(children_[i].load());
        for (uint16_t j = 0; j < leaf->size_; j++) {
          if (leaf->IsReadable(j)) {
            kvps.emplace_back(std::pair<KeyType, ValueType>(leaf->keys_[j], leaf->values_[j]));
          }
        }
      }

      sort(kvps.begin(), kvps.end(), [&](std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
        return this->tree_->KeyCmpLess(a.first, b.first);
      });

      // we put into each new leaf the min of (1) ceiling(total pairs / branch factor), (2) LEAF_SIZE / 2
      uint16_t optimal_leaf_size =
          (kvps.size() + static_cast<uint64_t>(BRANCH_FACTOR) - 1) / static_cast<uint64_t>(BRANCH_FACTOR);
      if (optimal_leaf_size < LEAF_OPTIMAL_FILL) {
        optimal_leaf_size = LEAF_OPTIMAL_FILL;
      }
      TERRIER_ASSERT(optimal_leaf_size <= LEAF_SIZE, "we should never have more than LEAF_SIZE many pairs in a leaf");
      TERRIER_ASSERT(optimal_leaf_size * BRANCH_FACTOR >= kvps.size(),
                     "we should have at least enough slots to cover all our tuples");

      auto *new_node = new InnerNode(this->tree_);
      auto *new_leaf = new LeafNode(this->tree_);
      uint16_t i;
      for (i = 0; i < optimal_leaf_size && i < kvps.size(); i++) {
        new_leaf->keys_[i] = kvps[i].first;
        new_leaf->values_[i] = kvps[i].second;
      }
      new_leaf->left_ = static_cast<LeafNode *>(children_[0].load())->left_.load();
      if (new_leaf->left_ != nullptr) {
        new_leaf->left_.load()->right_ = new_leaf;
      }
      new_leaf->size_ = i;
      new_node->children_[0] = new_leaf;

      uint64_t allocated_index = i;
      uint64_t inner_node_index = 0;
      while (allocated_index < kvps.size()) {
        TERRIER_ASSERT(inner_node_index < BRANCH_FACTOR, "must have at most branch factor many children");
        new_leaf = new LeafNode(this->tree_);
        for (i = 0; i < optimal_leaf_size && allocated_index + static_cast<uint64_t>(i) < kvps.size(); i++) {
          new_leaf->keys_[i] = kvps[i + allocated_index].first;
          new_leaf->values_[i] = kvps[i + allocated_index].second;
        }
        new_leaf->size_ = i;
        new_node->children_[inner_node_index + 1] = new_leaf;
        new_node->keys_[inner_node_index] = new_leaf->keys_[0];

        new_leaf->left_ = static_cast<LeafNode *>(new_node->children_[inner_node_index].load());
        static_cast<LeafNode *>(new_node->children_[inner_node_index].load())->right_ = new_leaf;

        allocated_index += static_cast<uint64_t>(i);
        inner_node_index++;
      }
      TERRIER_ASSERT(allocated_index == kvps.size(), "index should be equal to size");
      new_node->size_ = inner_node_index;

      new_leaf->right_ = static_cast<LeafNode *>(children_[this->size_].load())->right_.load();
      if (new_leaf->right_ != nullptr) {
        new_leaf->right_.load()->left_ = new_leaf;
      }

      return new_node;
    }

    std::atomic<BaseNode *> children_[BRANCH_FACTOR];
    KeyType keys_[BRANCH_FACTOR - 1];
  };

  /*
   * LeafNode: is a subclass of BaseNode. All the instances in the tree create a linked list of nodes along the bottom
   * of the tree. It contains LEAF_NODE_SIZE many key value pairs. Pairs are not sorted inside of leaves. Deleted paris
   * are marked as tomb stoned using a bitmap.
   */
  class LeafNode : public BaseNode {
   public:
    LeafNode() = default;

    explicit LeafNode(BPlusTree *tree) : BaseNode(tree, NodeType::LEAF), left_(nullptr), right_(nullptr) {}
    ~LeafNode() = default;

    /// ScanPredicate scans from this node to the right and evauates the given predicate on each equal key
    /// \param key
    /// \param predicate
    /// \return bool representing whether the predicate returned true on any key in this node or any node to the right
    /// of this node
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

    /// InsertLeaf inserts given key and value into this node. Will only do so if the given predicate returns false on
    /// all pairs with equal keys. Returns a bool indicating whether the insert succeded
    /// \param key
    /// \param value
    /// \param predicate
    /// \param predicate_satisfied return parameter of whether the predicate returned true on a equal key or not
    /// \return bool indicating whether the insert succeded
    OptimisticResult InsertLeaf(KeyType key, ValueType value, std::function<bool(const ValueType)> predicate,
                                bool *predicate_satisfied) {
      typename BaseNode::ScopedWriteLatch l(this);
      if (UNLIKELY(this->deleted_)) {
        return OptimisticResult::RetryableFailure;
      }

      *predicate_satisfied = false;
      bool saw_bigger = false;
      for (uint16_t i = 0; i < this->size_; i++) {
        if (IsReadable(i)) {
          if (this->tree_->KeyCmpEqual(key, keys_[i]) && predicate(values_[i])) {
            *predicate_satisfied = true;
            return OptimisticResult::Failure;
          }
          if (this->tree_->KeyCmpGreater(keys_[i], key)) {
            saw_bigger = true;
          }
        }
      }

      LeafNode *right = right_;
      if (!saw_bigger && right != nullptr && (*predicate_satisfied = right->ScanPredicate(key, predicate))) {
        return OptimisticResult::Failure;
      }

      // TODO(deepayan): see if insert to the right works out instead of forcing split here
      if (this->size_ >= this->GetLimit()) {
        return OptimisticResult::Failure;
      }

      // look for deleted slot
      uint16_t i;
      for (i = 0; i < this->size_ && IsReadable(i); i++) {
      }
      if (i != this->size_) {
        keys_[i] = key;
        values_[i] = value;
        UnmarkTombStone(i);
        return OptimisticResult::Success;
      }

      keys_[this->size_] = key;
      values_[this->size_] = value;
      UnmarkTombStone(this->size_);
      this->size_++;
      return OptimisticResult::Success;
    }

    /// ScanRange collects all values corresponding to keys between lo and hi where predicate returns true on the pair
    /// \param low
    /// \param hi
    /// \param values return parameter in which result values are added
    /// \param predicate
    /// \return bool indicating whether the range ends in this node or could continue in the next node
    bool ScanRange(KeyType low, KeyType *hi, std::vector<ValueType> *values,
                   std::function<bool(std::pair<KeyType, ValueType>)> predicate) {
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

      sort(temp_values.begin(), temp_values.end(),
           [&](std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
             return this->tree_->KeyCmpLess(a.first, b.first);
           });

      for (uint16_t i = 0; i < temp_values.size(); i++) {
        values->emplace_back(temp_values[i].second);
      }

      return res;
    }

    /// ScanRange collects all values corresponding to keys between lo and hi where predicate returns true on the value
    /// \param low
    /// \param hi
    /// \param values return parameter in which result values are added
    /// \param predicate
    /// \return bool indicating whether the range ends in this node or could continue in the next node
    bool ScanRangeReverse(KeyType low, KeyType *hi, std::vector<ValueType> *values,
                          std::function<bool(ValueType)> predicate) {
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

      sort(temp_values.begin(), temp_values.end(),
           [&](std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
             return this->tree_->KeyCmpGreater(a.first, b.first);
           });

      for (uint16_t i = 0; i < temp_values.size(); i++) {
        if (predicate(temp_values[i].second)) values->emplace_back(temp_values[i].second);
      }

      return res;
    }

    // TODO(emmanuel) use better bit ops???
    /// IsReadable returns true if the key value pair at the given slot is readable or is tombstoned
    /// \param i index of pair
    /// \return bool representing wheter pair is readable or tombstoned
    bool IsReadable(uint16_t i) {
      return !static_cast<bool>(
          (tomb_stones_[static_cast<uint64_t>(i) / BITS_IN_UINT64] >> (static_cast<uint64_t>(i) % BITS_IN_UINT64)) &
          static_cast<uint64_t>(0x1));
    }

    /// MarkTombStone marks the pair at the given index as deleted
    /// \param i index of pair
    void MarkTombStone(uint16_t i) {
      while (true) {
        uint64_t old_value = tomb_stones_[static_cast<uint64_t>(i) / BITS_IN_UINT64];
        uint16_t new_value = old_value | (static_cast<uint64_t>(0x1) << (static_cast<uint64_t>(i) % BITS_IN_UINT64));
        if (!tomb_stones_[static_cast<uint64_t>(i) / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value))
          continue;
        return;
      }
    }

    /// UnmarkTombStone unmarks the pair at the given index as marked
    /// \param i index of pair
    void UnmarkTombStone(uint16_t i) {
      while (true) {
        uint64_t old_value = tomb_stones_[static_cast<uint64_t>(i) / BITS_IN_UINT64];
        uint16_t new_value = old_value & (~(static_cast<uint64_t>(0x1) << (static_cast<uint64_t>(i) % BITS_IN_UINT64)));
        if (!tomb_stones_[static_cast<uint64_t>(i) / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value))
          continue;
        return;
      }
    }

    std::atomic<LeafNode *> left_, right_;
    std::atomic<uint64_t> tomb_stones_[(LEAF_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64] = {};
    ValueType values_[LEAF_SIZE];
    KeyType keys_[LEAF_SIZE];
  };

  template <class T>
  class NodeAllocator {
    class AllocatorWrapper {
     public:
      AllocatorWrapper() : allocator_array_(allocator_.allocate(ALLOCATOR_ARRAY_SIZE)) {}

      ~AllocatorWrapper() { allocator_.deallocate(allocator_array_, ALLOCATOR_ARRAY_SIZE); }

      bool Allocate(uint64_t i) {
        while (true) {
          uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
          uint64_t new_value =
              allocated_masks_[i / BITS_IN_UINT64] | (static_cast<uint64_t>(0x1) << (i % BITS_IN_UINT64));
          uint64_t available = old_value >> (i % BITS_IN_UINT64);
          if (static_cast<bool>(available & static_cast<uint64_t>(0x1))) return false;
          if (!allocated_masks_[i / BITS_IN_UINT64].compare_exchange_strong(old_value, new_value)) continue;
          return true;
        }
      }

      bool IsAllocated(uint64_t i) {
        uint64_t old_value = allocated_masks_[i / BITS_IN_UINT64];
        return static_cast<bool>((old_value >> (i % BITS_IN_UINT64)) & static_cast<uint64_t>(0x1));
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

      std::atomic<uint64_t> allocated_masks_[(ALLOCATOR_ARRAY_SIZE + BITS_IN_UINT64 - 1) / BITS_IN_UINT64] = {};
      std::allocator<T> allocator_;
      T *allocator_array_;
    };

   public:
    void ReclaimOldNodes(uint64_t safe_to_delete_epoch) {
      for (AllocatorWrapper a : allocator_set_) {
        for (uint64_t i = 0; i < ALLOCATOR_ARRAY_SIZE; i++) {
          if (a.IsAllocated(i)) {
            T *node = a.allocator_array_ + i;
            if (node->deleted_ && node->deleted_epoch_ <= safe_to_delete_epoch) {
              a.Reclaim(i);
            }
          }
        }
      }
    }

    NodeAllocator() : tree_(nullptr) { allocator_set_.insert(AllocatorWrapper()); }

    explicit NodeAllocator(BPlusTree *tree) : tree_(tree) {
      tree_->structure_size_ +=
          ALLOCATOR_ARRAY_SIZE *
          (((sizeof(T) + common::Constants::CACHELINE_SIZE - 1) / common::Constants::CACHELINE_SIZE) *
           common::Constants::CACHELINE_SIZE);
      allocator_set_.insert(AllocatorWrapper());
    }

    ~NodeAllocator() {
      for (AllocatorWrapper a : allocator_set_) {
        delete a;
      }
    }

    T *NewNode() {
      for (AllocatorWrapper a : allocator_set_) {
        uint64_t i;
        for (i = 0; i < ALLOCATOR_ARRAY_SIZE && !a.Allocate(i); i++) {
        }
        if (i != ALLOCATOR_ARRAY_SIZE) {
          a.allocator_[i] = T(tree_);
          return a.allocator_array_ + i;
        }
      }

      T *new_node;
      tree_->structure_size_ +=
          ALLOCATOR_ARRAY_SIZE *
          (((sizeof(T) + common::Constants::CACHELINE_SIZE - 1) / common::Constants::CACHELINE_SIZE) *
           common::Constants::CACHELINE_SIZE);
      AllocatorWrapper a = AllocatorWrapper();
      a.Allocate(0);
      a.allocator_[0] = T(tree_);
      new_node = a.allocator_array_ + 0;
      allocator_set_.insert(a);
      return new_node;
    }

   private:
    BPlusTree *tree_;
    tbb::concurrent_unordered_set<AllocatorWrapper, tbb::tbb_hash<AllocatorWrapper>, std::equal_to<AllocatorWrapper>,
                                  tbb::tbb_allocator<AllocatorWrapper>>
        allocator_set_;
  };

  void ReclaimOldNodes() {
    uint64_t old_epoch = epoch_;
    uint64_t iter = 1;
    while (active_epochs_[(old_epoch + iter) % MAX_NUM_ACTIVE_EPOCHS] != 0) {
    }
    for (iter = 2; iter < MAX_NUM_ACTIVE_EPOCHS - 1 && active_epochs_[(old_epoch + iter) % MAX_NUM_ACTIVE_EPOCHS] == 0;
         iter++) {
    }

//    uint64_t safe_iter = iter - 1;
//    uint64_t safe_to_delete_epoch = old_epoch + safe_iter - MAX_NUM_ACTIVE_EPOCHS;

    epoch_++;

//    inner_node_allocator_.ReclaimOldNodes(safe_to_delete_epoch);
//    leaf_node_allocator_.ReclaimOldNodes(safe_to_delete_epoch);
  }

  void RunGarbageCollection() {
    gc_latch_.Lock();
//    CompressTree();
    ReclaimOldNodes();
    gc_latch_.Unlock();
  }

  /// StartFunction records that a function was started. Returns the epoch in which the function was started
  /// \return the epoch in which the function was started
  uint64_t StartFunction() {
    uint64_t current_epoch = epoch_;
    active_epochs_[current_epoch % MAX_NUM_ACTIVE_EPOCHS]++;
    return current_epoch;
  }

  /// EndFunction marks that a function that was started at the given epoch has ended
  /// \param start_epoch epoch in which the function was started
  void EndFunction(uint64_t start_epoch) { active_epochs_[start_epoch % MAX_NUM_ACTIVE_EPOCHS]--; }

  /// OptimisticInsert inserts under the assumption that the leaf node into which the pair is inserted will not split
  /// evaluates the predicate on all values that have equal keys
  /// \param predicate predicate to be evaluated on all equal keys
  /// \param key
  /// \param val
  /// \param predicate_satisfied bool representing whether the predicate returned true on the value of an equal key
  /// \return bool representing whether the insert succeeded
  bool OptimisticInsert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val,
                        bool *predicate_satisfied) {
    OptimisticResult result = FindMinLeaf(key)->InsertLeaf(key, val, predicate, predicate_satisfied);
    while (result == OptimisticResult::RetryableFailure) {
      result = FindMinLeaf(key)->InsertLeaf(key, val, predicate, predicate_satisfied);
    }
    return (result == OptimisticResult::Success);
  }

  /// InsertHelper Helper function for insert. Actually implements the insertion. Calls Optimistic insert. If
  /// OptimisticInsert insert fails it takes write locks down the tree using lock crabbing and preforms a split leaf
  /// node and all necessary ancestors.
  /// \param predicate must return false values associated with all equal keys for insert to succeed
  /// \param key
  /// \param val
  /// \param predicate_satisfied return value indicating whether the predicate returned true on values associated with
  /// equal keys
  /// \return bool indicating whether the insert succeeded
  bool InsertHelper(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val,
                    bool *predicate_satisfied) {
    if (OptimisticInsert(predicate, key, val, predicate_satisfied)) {
      return true;
    }

    std::vector<InnerNode *> locked_nodes;
    std::vector<uint16_t> traversal_indices;
    bool holds_tree_latch;

    LatchRoot();
    holds_tree_latch = true;
    BaseNode *n = this->root_;
    TERRIER_ASSERT(!n->deleted_, "we should never traverse deleted nodes");

    // Find minimum leaf that stores would store key, taking locks as we go down tree
    while (n->GetType() != NodeType::LEAF) {
      auto inner_n = static_cast<InnerNode *>(n);
      inner_n->Lock();
      if (n->size_ < n->GetLimit()) {
        if (holds_tree_latch && locked_nodes.size() > 1) {
          UnlatchRoot();
          holds_tree_latch = false;
        }
        for (uint64_t i = 0; (!locked_nodes.empty()) && i < locked_nodes.size() - 1; i++) {
          locked_nodes[i]->Unlock();
        }
        if (!locked_nodes.empty()) {
          InnerNode *last = locked_nodes.back();
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

    // If leaf is not full, insert, unlock all parent nodes and return
    auto leaf = static_cast<LeafNode *>(n);
    if (leaf->InsertLeaf(key, val, predicate, predicate_satisfied) == OptimisticResult::Success ||
        *predicate_satisfied) {
      if (holds_tree_latch) {
        UnlatchRoot();
        holds_tree_latch = false;
      }
      for (InnerNode *node : locked_nodes) node->Unlock();

      return !(*predicate_satisfied);
    }

    // returned false because of no tomb stones;

    LeafNode *left, *right;
    while (true) {
      left = leaf->left_.load();
      right = leaf->right_.load();
      if (LIKELY(left != nullptr)) {
        left->Lock();
        if (UNLIKELY(left != leaf->left_ || left->deleted_)) {
          left->Unlock();
          continue;
        }
      }
      leaf->Lock();
      if (LIKELY(right != nullptr)) {
        right->Lock();
        if (UNLIKELY(right != leaf->right_ || right->deleted_)) {
          if (LIKELY(left != nullptr)) {
            left->Unlock();
          }
          leaf->Unlock();
          right->Unlock();
          continue;
        }
      }
      break;
    }

    TERRIER_ASSERT(!leaf->deleted_ && (left == nullptr || !left->deleted_) && (right == nullptr || !right->deleted_),
                   "none of leaf right and left should be marked as deleted");

    // Otherwise must split so create new leaf
    LeafNode *new_leaf_right = new LeafNode(this);
    LeafNode *new_leaf_left = new LeafNode(this);

    std::vector<std::pair<KeyType, ValueType>> kvps;
    kvps.emplace_back(std::pair<KeyType, ValueType>(key, val));
    for (uint16_t i = 0; i < leaf->size_; i++) {
      kvps.emplace_back(std::pair<KeyType, ValueType>(leaf->keys_[i], leaf->values_[i]));
    }

    sort(kvps.begin(), kvps.end(), [&](std::pair<KeyType, ValueType> a, std::pair<KeyType, ValueType> b) {
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
      left->Unlock();
    }
    leaf->MarkDeleted();
    leaf->Unlock();
    if (LIKELY(right != nullptr)) {
      right->left_ = new_leaf_right;
      right->Unlock();
    }

    // If split is on root (leaf)
    if (UNLIKELY(locked_nodes.empty())) {
      TERRIER_ASSERT(leaf == root_, "we had to split a leaf without having to modify the parent");
      TERRIER_ASSERT(holds_tree_latch, "we are trying to split the root without a latch on the tree");
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
      if (old_node->size_ < old_node->GetLimit()) {
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
      old_node->Unlock();
    }

    if (old_node->size_ < old_node->GetLimit()) {
      //      TERRIER_ASSERT(old_node->write_latch_, "must hold write latch if not full");
      InnerNode *new_node = new InnerNode(this);
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
      old_node->Unlock();

      if (old_node == root_) {
        TERRIER_ASSERT(holds_tree_latch && locked_nodes.empty(), "must hold tree latch if held latch on root");
        root_ = new_node;
        UnlatchRoot();
        holds_tree_latch = false;
      } else {
        TERRIER_ASSERT(!holds_tree_latch || !locked_nodes.empty(),
                       "if we are not at the root then we should not hold the tree latch"
                       "and should have released some write latches");
        child_index = traversal_indices.back();
        traversal_indices.pop_back();
        old_node = locked_nodes.back();
        locked_nodes.pop_back();
        num_popped++;
        old_node->children_[child_index] = new_node;
        old_node->Unlock();
        if (holds_tree_latch) {
          TERRIER_ASSERT(old_node == root_, "if we hold the tree latch, then we are adjusting the root");
          UnlatchRoot();
          holds_tree_latch = false;
        }
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
      UnlatchRoot();
      holds_tree_latch = false;
    }

    TERRIER_ASSERT(locked_nodes.size() == 0, "we should have unlocked all locked nodes");
    TERRIER_ASSERT(!holds_tree_latch, "should not hold the rootlatch on return");
    return true;
  }

  /// Insert inserts given key value pair into tree if the predicate provided returns true on value associated with key
  /// \param predicate must return false values associated with all equal keys for insert to succeed
  /// \param key
  /// \param val
  /// \param predicate_satisfied return value indicating whether the predicate returned true on values associated with
  /// \return bool indicating whether the insert succeeded
  bool Insert(std::function<bool(const ValueType)> predicate, KeyType key, ValueType val, bool *predicate_satisfied) {
    uint64_t epoch = StartFunction();
    bool result = InsertHelper(predicate, key, val, predicate_satisfied);
    EndFunction(epoch);
    return result;
  }

  /// ScanKeyHelper helper function for ScanKey. Returns vector of all values associated with the given key on which
  /// the given predicate returns true
  /// \param key
  /// \param values return vector of values associated with key on which the predicate returns true
  /// \param predicate
  void ScanKeyHelper(KeyType key, std::vector<ValueType> *values, std::function<bool(ValueType)> predicate) {
    bool done = false;
    for (auto *leaf = FindMinLeaf(key); leaf != nullptr && !done; leaf = leaf->right_) {
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

  /// ScanKey Returns vector of all values associated with the given key on which
  /// the given predicate returns true
  /// \param key
  /// \param values return vector of values associated with key on which the predicate returns true
  /// \param predicate
  void ScanKey(
      KeyType key, std::vector<ValueType> *values,
      std::function<bool(ValueType)> predicate = [](ValueType v) { return true; }) {
    uint64_t epoch = StartFunction();
    ScanKeyHelper(key, values, predicate);
    EndFunction(epoch);
  }

  /// RemoveHelper Helper function for Remove. Removes key value pair given
  /// \param key
  /// \param value
  /// \return bool representing whether pair was successful
  bool RemoveHelper(KeyType key, ValueType value) {
    bool done = false;
    while (true) {
    Loop:
      for (LeafNode *leaf = FindMinLeaf(key); leaf != nullptr && !done; leaf = leaf->right_) {
        typename BaseNode::ScopedWriteLatch l(leaf);
        if (leaf->deleted_) {
          goto Loop;
        }
        for (uint16_t i = 0; i < leaf->size_; i++) {
          if (leaf->IsReadable(i)) {
            if (KeyCmpEqual(key, leaf->keys_[i]) && value_eq_obj_(value, leaf->values_[i])) {
              leaf->MarkTombStone(i);
              return true;
            }
            if (KeyCmpLess(key, leaf->keys_[i])) {
              done = true;
            }
          }
        }
      }
    }
    return false;
  }

  /// Remove Removes key value pair given
  /// \param key
  /// \param value
  /// \return bool representing wheter pair was successfull
  bool Remove(KeyType key, ValueType value) {
    uint64_t epoch = StartFunction();
    bool result = RemoveHelper(key, value);
    EndFunction(epoch);
    return result;
  }

  /// MergeToDepth Recurses on all subchildren until the given node is at the max depth. Then merges and replaces all
  /// chilren of the given node.
  /// \param node
  /// \param max_depth max depth to be reached before merging
  /// \param current_depth current depth in the tree
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
        if (UNLIKELY(node->children_[i].load()->deleted_ || node->children_[i].load()->GetType() == NodeType::LEAF))
          continue;
        auto *child = static_cast<InnerNode *>(node->children_[i].load());
        node->children_[i] = child->Merge();
      }
      return;
    }

    for (uint16_t i = 0; i <= node->size_; i++) {
      MergeToDepth(static_cast<InnerNode *>(node->children_[i].load()), max_depth, current_depth + 1);
    }
  }

  /// CompressTree compresses the given tree my merging all mergable nodes. Does so by iteative shallowing and merging
  /// from the bottom up. As a result, the tree is able to be compressed without blocking inserts or reads.
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
    for (uint64_t d = depth - 2; d >= 1; d--) {
      MergeToDepth(static_cast<InnerNode *>(root_.load()), d, 1);
      std::this_thread::sleep_for (std::chrono::seconds (1));
    }

    if (LIKELY(root_.load()->GetType() == NodeType::INNER_NODE)) {
      LatchRoot();
      root_ = static_cast<InnerNode *>(root_.load())->Merge();
      UnlatchRoot();
    }

  }

  /// GetDepth retuns the depth of the tree
  /// \return
  uint64_t GetDepth() {
    uint64_t d = 1;
    BaseNode *n = root_;
    while (n->GetType() != NodeType::LEAF) {
      n = static_cast<InnerNode *>(n)->children_[0];
      d++;
    }
    return d;
  }

  /// FindMinLeaf finds the minimum leaf in the tree
  /// \return pointer to the right most leaf of the tree
  LeafNode *FindMinLeaf() {
    while (true) {
    OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        auto *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->children_[0];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  /// FindMinLeaf finds the minimum leaf in the tree that could contain the given key
  /// \return pointer to the right most leaf of the tree that could contain the given key
  LeafNode *FindMinLeaf(KeyType key) {
    while (true) {
    OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        auto *inner_n = static_cast<InnerNode *>(n);
        uint16_t child_index = inner_n->FindMinChild(key);
        n = inner_n->children_[child_index];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  /// FindMaxLeaf finds the maximum leaf in the tree
  /// \return pointer to the left most leaf of the tree
  LeafNode *FindMaxLeaf() {
    while (true) {
    OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        auto *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->children_[n->size_];
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  /// FindMaxLeaf finds the maximum leaf in the tree that could contain the given key
  /// \return pointer to the left most leaf of the tree that could contain the given key
  LeafNode *FindMaxLeaf(KeyType key) {
    while (true) {
    OuterLoop:
      BaseNode *n = root_;
      if (UNLIKELY(n->deleted_)) {
        goto OuterLoop;
      }
      while (n->GetType() != NodeType::LEAF) {
        auto *inner_n = static_cast<InnerNode *>(n);
        n = inner_n->FindMaxChild(key);
        if (n->deleted_) {
          goto OuterLoop;
        }
      }
      return static_cast<LeafNode *>(n);
    }
  }

  /// LatchRoot locks modificiation the root. Does not block reading
  void LatchRoot() {
    //    bool t = true;
    //    bool f = false;
    //    while (!root_latch_.compare_exchange_strong(f, t)) {}
    //    TERRIER_ASSERT(root_latch_, "root latch should be held");
    root_latch_.Lock();
  }

  /// UnlatchRoot unlocks modificiation the root
  void UnlatchRoot() {
    //    TERRIER_ASSERT(root_latch_, "root latch should be held");
    //    root_latch_ = false;
    root_latch_.Unlock();
  }

  // Key comparator, and key and value equality checker
  KeyComparator key_cmp_obj_;
  KeyEqualityChecker key_eq_obj_;
  ValueEqualityChecker value_eq_obj_;
  std::atomic<uint64_t> active_epochs_[MAX_NUM_ACTIVE_EPOCHS] = {};
  std::atomic<uint64_t> structure_size_ = 5;
  std::atomic<uint64_t> epoch_ = 1;
  std::atomic<BaseNode *> root_;
//  NodeAllocator<InnerNode> inner_node_allocator_ = NodeAllocator<InnerNode>(this);
//  NodeAllocator<LeafNode> leaf_node_allocator_ = NodeAllocator<LeafNode>(this);
  common::SpinLatch root_latch_, gc_latch_;
};

}  // namespace terrier::storage::index
