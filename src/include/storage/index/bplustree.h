#pragma once

#include <functional>
#include "common/macros.h"

namespace terrier::storage::index {

template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
    typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
    typename ValueEqualityChecker = std::equal_to<ValueType>>
class BPlusTree {
  class InnerNode;
  class Leaf;
  enum class NodeType: bool;

  class GeneralNode;

  enum class NodeType: bool {
    LEAF = true,
    INNER_NODE = false,
  };

  class GeneralNode {
   public:
    GeneralNode(NodeType t);
    ~GeneralNode();

    inline void write_lock() {
      while(true) {
        uint64_t old_value = meta_data_.load();
        uint64_t not_locked = old_value | (~write_lock_mask_);
        uint64_t locked = old_value | write_lock_mask_;
        if (meta_data_.compare_exchange_strong(not_locked, locked)) {
          continue;
        }
        break;
      }
    };

    inline void write_unlock() {
      TERRIER_ASSERT(meta_data_.load() & write_lock_mask_, "unlock called on unlocked inner node");
      meta_data_ = meta_data_.load() | (~write_lock_mask_);
    };

    inline void mark_delete() {
      TERRIER_ASSERT(meta_data_.load() & is_deleted_mask_,
                     "tried to mark as deleted already deleted node");
      meta_data_ = meta_data_.load() | (~is_deleted_mask_);
    }

    inline NodeType get_type() { return static_cast<NodeType>(meta_data_ & node_type_mask_); }

   private:
    std::atomic<uint64_t> meta_data_;
    char* node_;
  };

  class InnerNode {
   public:
    InnerNode();
    ~InnerNode();


   private:
    friend class BPlusTree;

  };

  class Leaf {
   public:
    Leaf();
    ~Leaf();

   private:
    friend class BPlusTree;

  };

 public:
  BPlusTree();
  ~BPlusTree();

 protected:
  // ideally running with 64 byte cache line
  std::atomic<uint64_t> epoch = 0;
  const short cache_line_size_ = 64;
  const short leaf_node_size_ = cache_line_size_;
  const short inner_node_size_ = cache_line_size_;
  static const uint64_t write_lock_mask_ = static_cast<uint64_t>(0x01) << 63;
  static const uint64_t node_type_mask_ = static_cast<uint64_t>(0x01) << 62;
  static const uint64_t is_deleted_mask_ = static_cast<uint64_t>(0x01) << 61;




};

}  // namespace terrier::storage::index