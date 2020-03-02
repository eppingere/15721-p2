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
      : root_{static_cast<BaseNode *>(new LeafNode)}, size_{sizeof(LeafNode)} {};

  static std::atomic<uint64_t> epoch;
  static const short branch_factor_ = 8;
  static const short leaf_size_ = 8;
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
      if (this->size_.load() == 0) {
        return NULL;
      } else {
        uint16_t i;
        for (i = 0; i < this->size_.load(); i++)
          if (KeyCmpLessEqual(key, keys_[i]))
            return children_[i];

        return children_[i-1];
      }
    }

    BaseNode* findMaxChild(KeyType key) {
      if (this->size_.load() == 0) {
        return NULL;
      } else {
        uint16_t i;
        for (i = this->size_.load() - 1; i >= 0; i--)
          if (KeyCmpGreaterEqual(key, keys_[i]))
            return children_[i];

        return children_[i];
      }
    }

    std::atomic<BaseNode *> children_[branch_factor_];
    KeyType keys_[branch_factor_ - 1];
  };

  class LeafNode : public BaseNode {
   public:
    static const uint64_t delete_mask_size_ = 64;

    LeafNode() : BaseNode(NodeType::LEAF), left_(nullptr), right_(nullptr) {};
    ~LeafNode()=default;

    inline bool take_toumb_stone_slot(uint16_t i) {
      if (UNLIKELY(i >= toumb_stones_length_)) return false;
      uint64_t index = i / delete_mask_size_;
      uint64_t mask = static_cast<uint64_t>(0x1) << (i % delete_mask_size_);
      uint64_t old_val = toumb_stones_[index].load();
      return static_cast<bool>(old_val & mask) &&
             toumb_stones_[index].compare_exchange_strong(old_val, old_val & (~mask));
    }

    inline void mark_toumb_stone(uint16_t i) {
      if (UNLIKELY(i >= toumb_stones_length_)) return;
      uint64_t index = i / delete_mask_size_;
      uint64_t mask = static_cast<uint64_t>(0x1) << (i % delete_mask_size_);
      while (true) {
        uint64_t old_val = toumb_stones_[index].load();
        if (!toumb_stones_[index].compare_exchange_strong(old_val, old_val | mask)) continue;
        break;
      }
    }

    inline bool is_occupied(uint16_t i) {
      if (UNLIKELY(i >= toumb_stones_length_)) return false;
      uint64_t index = i / delete_mask_size_;
      uint64_t mask = static_cast<uint64_t>(0x1) << (i % delete_mask_size_);
      uint64_t val = toumb_stones_[index].load();
      return !static_cast<bool>(val & mask);
    }

    bool insert(KeyType key, ValueType value) {
      // look for deleted slot
      for (uint16_t i = 0; i < this->size_; i++) {
        if (take_toumb_stone_slot(i)) {
          keys_[i] = key;
          values_[i] = value;
          return true;
        }
      }

      uint16_t key_offset;
      while (true) {
        key_offset = this->offset_.load();
        if (!this->offset_.compare_exchange_strong(key_offset, key_offset + 1)) {
          continue;
        }
        break;
      }

      if (key_offset >= this->limit_) {
        return false;
      }

      keys_[key_offset] = key;
      values_[key_offset] = value;
      while (this->size_.load() < key_offset) {}
      this->size_++;
      return true;
    }

    bool mark_delete(KeyType key, ValueType value) {
      for (uint16_t i = 0; i < size_.load(); i++)
        if (is_occupied(i) && KeyCmpEqual(key, keys_[i]) && ValueCmpEqual(value, values_[i])) {
          mark_toumb_stone(i);
          return true;
        }
      return false;
    }

    bool scan_range(KeyType low, KeyType hi, std::vector<ValueType> *values) {
      bool res = true;
      for (uint16_t i = 0; i < size_.load(); i++) {
        if (is_occupied(i) && KeyCmpGreaterEqual(keys_[i], low) && KeyCmpLessEqual(keys_[i], hi))
          values->emplace_back(values_[i]);
        else if (KeyCmpGreater(keys_[i], hi))
          res = false;
      }
      return res;
    }

    static const uint64_t toumb_stones_length_ = (leaf_size_ + (delete_mask_size_ - 1)) / delete_mask_size_;
    std::atomic<uint64_t> toumb_stones_[toumb_stones_length_];
    std::atomic<LeafNode*> left_, right_;
    KeyType keys_[leaf_size_];
    ValueType values_[leaf_size_];
  };

  inline bool KeyCmpLess(const KeyType &key1, const KeyType &key2) const { return key_cmp_obj(key1, key2); }

  inline bool KeyCmpEqual(const KeyType &key1, const KeyType &key2) const { return key_eq_obj(key1, key2); }

  inline bool KeyCmpGreaterEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpLess(key1, key2); }

  inline bool KeyCmpGreater(const KeyType &key1, const KeyType &key2) const { return KeyCmpLess(key2, key1); }

  inline bool KeyCmpLessEqual(const KeyType &key1, const KeyType &key2) const { return !KeyCmpGreater(key1, key2); }

  inline bool ValueCmpEqual(const ValueType &v1, const ValueType &v2) { return value_eq_obj(v1, v2); }

  BaseNode *root_;
  std::atomic<uint64_t> size_;

  // TODO: Check if bool or bool (*) is correct type of predicate
  bool InsertIf(std::function<bool (const ValueType)> predicate, KeyType key, ValueType value);
  LeafNode* FindMin(KeyType key);
  LeafNode* FindMax(KeyType key);
  void FindRange(KeyType min_key, KeyType max_key, bool is_increasing, std::vector <ValueType> *results);
};

}  // namespace terrier::storage::index