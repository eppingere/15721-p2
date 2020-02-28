#pragma once

#include <common/shared_latch.h>

#include <functional>

#include "common/macros.h"

namespace terrier::storage::index {

template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
    typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
    typename ValueEqualityChecker = std::equal_to<ValueType>>
class BPlusTree {

  static std::atomic<uint64_t> epoch;
  static const short branch_factor_ = 8;
  static const short leaf_size_ = 8;
  static const uint64_t write_lock_mask_ = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t node_type_mask_ = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t is_deleted_mask_ = static_cast<uint64_t>(0x01) << 61;
  static const uint64_t delete_epoch_mask_ = ~(static_cast<uint64_t>(0xE) << 60) & (~static_cast<uint64_t>(0xF));
  static const uint64_t size_mask_ = static_cast<uint64_t>(0xF);
  static const uint64_t size_length_ = static_cast<uint64_t>(8);

  enum class NodeType: bool {
    LEAF = true,
    INNER_NODE = false,
  };

  class BaseNode {
   public:
    BaseNode(BPlusTree::NodeType t) : info_(t == NodeType::LEAF ? node_type_mask_ : 0) {}
    ~BaseNode()=default;

    std::atomic<uint64_t> info_;
    common::SharedLatch base_latch_;

    inline void write_lock() {
      while(true) {
        uint64_t old_value = info_.load();
        uint64_t not_locked = old_value | (~write_lock_mask_);
        uint64_t locked = old_value | write_lock_mask_;
        if (info_.compare_exchange_strong(not_locked, locked)) {
          continue;
        }
        break;
      }
    };

    inline void write_unlock() {
      TERRIER_ASSERT(info_.load() & write_lock_mask_, "unlock called on unlocked inner node");
      while (true) {
        uint64_t old_meta_data = info_.load();
        if (info_.compare_exchange_strong(old_meta_data, old_meta_data | (~write_lock_mask_))) {
          continue;
        }
        break;
      }
    };

    inline void mark_delete() {
      TERRIER_ASSERT(!(info_.load() & is_deleted_mask_),
                     "tried to mark as deleted already deleted node");
      while (true) {
        uint64_t old_meta_data = info_.load();
        uint64_t e = epoch.load() & delete_epoch_mask_;
        if (info_.compare_exchange_strong(old_meta_data, old_meta_data | (~is_deleted_mask_) | e)) {
          continue;
        }
        break;
      }
    }

    inline uint64_t get_delete_epoch() {
      TERRIER_ASSERT(info_.load() & is_deleted_mask_, "got delete epoch of non deleted node");
      return (info_.load() & delete_epoch_mask_) >> size_length_;
    }

    inline NodeType get_type() { return static_cast<NodeType>(info_ & node_type_mask_); }

    inline NodeType get_size() { return info_ & size_mask_; }
  };

  class InnerNode : BaseNode {
   public:
    InnerNode() : BaseNode{NodeType::INNER_NODE} {},
    ~InnerNode()=default;

    std::atomic<InnerNode *> children_[branch_factor_];
    KeyType keys_[branch_factor_ - 1];
  };

  class LeafNode : BaseNode {
   public:
    LeafNode() : BaseNode{NodeType::LEAF} {};
    ~LeafNode()=default;

    std::atomic<LeafNode*> left, right;
    KeyType keys_[leaf_size_];
    ValueType values_[leaf_size_];
  };

};

}  // namespace terrier::storage::index