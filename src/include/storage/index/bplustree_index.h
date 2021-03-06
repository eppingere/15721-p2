#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "storage/index/bplustree.h"
#include "storage/index/index.h"
#include "storage/index/index_defs.h"
#include "transaction/deferred_action_manager.h"
#include "transaction/transaction_context.h"
#include "transaction/transaction_manager.h"

namespace terrier::storage::index {
template <uint8_t KeySize>
class CompactIntsKey;
template <uint16_t KeySize>
class GenericKey;

/**
 * Wrapper around 15-721 Project 2's B+Tree
 * @tparam KeyType the type of keys stored in the BPlusTree
 */
template <typename KeyType>
class BPlusTreeIndex final : public Index {
  friend class IndexBuilder;

 private:
  explicit BPlusTreeIndex(IndexMetadata metadata)
      : Index(std::move(metadata)), bplustree_{new BPlusTree<KeyType, TupleSlot>} {}

  const std::unique_ptr<BPlusTree<KeyType, TupleSlot>> bplustree_;

 public:
  IndexType Type() const final { return IndexType::BPLUSTREE; }

  void PerformGarbageCollection() final {
    // FIXME(15-721 project2): invoke garbage collection on the underlying data structure
    bplustree_->RunGarbageCollection();
  }

  size_t GetHeapUsage() const final {
    // FIXME(15-721 project2): access the underlying data structure and report the heap usage
    return bplustree_->structure_size_;
  }

  bool Insert(const common::ManagedPointer<transaction::TransactionContext> txn, const ProjectedRow &tuple,
              const TupleSlot location) final {
    TERRIER_ASSERT(!(metadata_.GetSchema().Unique()),
                   "This Insert is designed for secondary indexes with no uniqueness constraints.");
    KeyType index_key;
    index_key.SetFromProjectedRow(tuple, metadata_, metadata_.GetSchema().GetColumns().size());
    // FIXME(15-721 project2): perform a non-unique unconditional insert into the underlying data structure of the
    // key/value pair

    auto predicate UNUSED_ATTRIBUTE = [](const TupleSlot slot) -> bool { return false; };
    bool predicate_satisfied = false;

    const bool UNUSED_ATTRIBUTE result = bplustree_->Insert(index_key, location, &predicate_satisfied);

    TERRIER_ASSERT(
        result && !predicate_satisfied,
        "non-unique index shouldn't fail to insert. If it did, something went wrong deep inside the BPlusTree itself.");
    // Register an abort action with the txn context in case of rollback
    txn->RegisterAbortAction([=]() {
      // FIXME(15-721 project2): perform a delete from the underlying data structure of the key/value pair
      const bool UNUSED_ATTRIBUTE result = bplustree_->Remove(index_key, location);

      TERRIER_ASSERT(result, "Delete on the index failed.");
    });
    return result;
  }

  bool InsertUnique(const common::ManagedPointer<transaction::TransactionContext> txn, const ProjectedRow &tuple,
                    const TupleSlot location) final {
    TERRIER_ASSERT(metadata_.GetSchema().Unique(), "This Insert is designed for indexes with uniqueness constraints.");
    KeyType index_key;
    index_key.SetFromProjectedRow(tuple, metadata_, metadata_.GetSchema().GetColumns().size());
    bool predicate_satisfied UNUSED_ATTRIBUTE = false;

    // The predicate checks if any matching keys have write-write conflicts or are still visible to the calling txn.
    auto predicate UNUSED_ATTRIBUTE = [txn](const TupleSlot slot) -> bool {
      const auto *const data_table = slot.GetBlock()->data_table_;
      const auto has_conflict = data_table->HasConflict(*txn, slot);
      const auto is_visible = data_table->IsVisible(*txn, slot);
      return has_conflict || is_visible;
    };

    // FIXME(15-721 project2): perform a non-unique CONDITIONAL insert into the underlying data structure of the
    // key/value pair
    const bool result = bplustree_->Insert(index_key, location, &predicate_satisfied, predicate);

    TERRIER_ASSERT(predicate_satisfied != result, "If predicate is not satisfied then insertion should succeed.");

    if (result) {
      // Register an abort action with the txn context in case of rollback
      txn->RegisterAbortAction([=]() {
        // FIXME(15-721 project2): perform a delete from the underlying data structure of the key/value pair
        const bool UNUSED_ATTRIBUTE result = bplustree_->Remove(index_key, location);
        TERRIER_ASSERT(result, "Delete on the index failed.");
      });
    } else {
      // Presumably you've already made modifications to a DataTable (the source of the TupleSlot argument to this
      // function) however, the index found a constraint violation and cannot allow that operation to succeed. For MVCC
      // correctness, this txn must now abort for the GC to clean up the version chain in the DataTable correctly.
      txn->SetMustAbort();
    }

    return result;
  }

  void Delete(const common::ManagedPointer<transaction::TransactionContext> txn, const ProjectedRow &tuple,
              const TupleSlot location) final {
    KeyType index_key;
    index_key.SetFromProjectedRow(tuple, metadata_, metadata_.GetSchema().GetColumns().size());

    TERRIER_ASSERT(!(location.GetBlock()->data_table_->HasConflict(*txn, location)) &&
                       !(location.GetBlock()->data_table_->IsVisible(*txn, location)),
                   "Called index delete on a TupleSlot that has a conflict with this txn or is still visible.");

    // Register a deferred action for the GC with txn manager. See base function comment.
    txn->RegisterCommitAction([=](transaction::DeferredActionManager *deferred_action_manager) {
      deferred_action_manager->RegisterDeferredAction([=]() {
        // FIXME(15-721 project2): perform a delete from the underlying data structure of the key/value pair
        const bool UNUSED_ATTRIBUTE result = bplustree_->Remove(index_key, location);

        TERRIER_ASSERT(result, "Deferred delete on the index failed.");
      });
    });
  }

  void ScanKey(const transaction::TransactionContext &txn, const ProjectedRow &key,
               std::vector<TupleSlot> *value_list) final {
    TERRIER_ASSERT(value_list->empty(), "Result set should begin empty.");

    // Build search key
    KeyType index_key;
    index_key.SetFromProjectedRow(key, metadata_, metadata_.GetSchema().GetColumns().size());

    // Perform lookup in BPlusTree
    // FIXME(15-721 project2): perform a lookup of the underlying data structure of the key

    std::vector<TupleSlot> results;

    // Perform lookup in BwTree
    bplustree_->ScanKey(index_key, &results);

    // Avoid resizing our value_list, even if it means over-provisioning
    value_list->reserve(results.size());

    // Perform visibility check on result
    for (const auto &result : results) {
      if (IsVisible(txn, result)) value_list->emplace_back(result);
    }

    TERRIER_ASSERT(!(metadata_.GetSchema().Unique()) || (metadata_.GetSchema().Unique() && value_list->size() <= 1),
                   "Invalid number of results for unique index.");
  }

  void ScanAscending(const transaction::TransactionContext &txn, ScanType scan_type, uint32_t num_attrs,
                     ProjectedRow *low_key, ProjectedRow *high_key, uint32_t limit,
                     std::vector<TupleSlot> *value_list) final {
    TERRIER_ASSERT(value_list->empty(), "Result set should begin empty.");
    TERRIER_ASSERT(scan_type == ScanType::Closed || scan_type == ScanType::OpenLow || scan_type == ScanType::OpenHigh ||
                       scan_type == ScanType::OpenBoth,
                   "Invalid scan_type passed into BPlusTreeIndex::Scan");

    bool low_key_exists = (scan_type == ScanType::Closed || scan_type == ScanType::OpenHigh);
    bool high_key_exists = (scan_type == ScanType::Closed || scan_type == ScanType::OpenLow);

    // Build search keys
    KeyType index_low_key, index_high_key;
    if (low_key_exists) index_low_key.SetFromProjectedRow(*low_key, metadata_, num_attrs);
    if (high_key_exists) index_high_key.SetFromProjectedRow(*high_key, metadata_, num_attrs);

    uint64_t epoch = bplustree_->StartFunction();

    std::function<bool(std::pair<KeyType, TupleSlot>)> predicate = [&](std::pair<KeyType, TupleSlot> p) {
      return IsVisible(txn, p.second) &&
             (!high_key_exists || p.first.PartialLessThan(index_high_key, &metadata_, num_attrs)) &&
             (!low_key_exists || index_low_key.PartialLessThan(p.first, &metadata_, num_attrs));
    };

    // FIXME(15-721 project2): perform a lookup of the underlying data structure of the key
    bool done = false;
    for (auto *leaf = low_key_exists ? bplustree_->FindMinLeafReadOnly(index_low_key)
                                     : bplustree_->FindMinLeafReadOnly();
         leaf != nullptr && (limit == 0 || value_list->size() < limit) && !done; leaf = leaf->right_) {
      for (uint16_t i = 0; i < leaf->size_ && (limit == 0 || value_list->size() < limit); i++) {
        if (leaf->IsReadable(i)) {
          if (predicate({leaf->keys_[i], leaf->values_[i]})) {
            value_list->emplace_back(leaf->values_[i]);
          } else if (high_key_exists && !leaf->keys_[i].PartialLessThan(index_high_key, &metadata_, num_attrs)) {
            done = true;
          }
        }
      }
    }

    bplustree_->EndFunction(epoch);
  }

  void ScanDescending(const transaction::TransactionContext &txn, const ProjectedRow &low_key,
                      const ProjectedRow &high_key, std::vector<TupleSlot> *value_list) final {
    TERRIER_ASSERT(value_list->empty(), "Result set should begin empty.");

    // Build search keys
    KeyType index_low_key, index_high_key;
    index_low_key.SetFromProjectedRow(low_key, metadata_, metadata_.GetSchema().GetColumns().size());
    index_high_key.SetFromProjectedRow(high_key, metadata_, metadata_.GetSchema().GetColumns().size());

    // FIXME(15-721 project2): perform a lookup of the underlying data structure of the key
    // Perform lookup in BwTree

    uint64_t epoch = bplustree_->StartFunction();

    std::function<bool(TupleSlot)> predicate = [&](TupleSlot tuple) { return IsVisible(txn, tuple); };

    // FIXME(15-721 project2): perform a lookup of the underlying data structure of the key
    for (auto *leaf = bplustree_->FindMaxLeafReadOnly(index_high_key); leaf != nullptr; leaf = leaf->left_) {
      if (!leaf->ScanRangeReverse(index_low_key, &index_high_key, value_list, predicate)) {
        break;
      }
    }

    bplustree_->EndFunction(epoch);
  }

  void ScanLimitDescending(const transaction::TransactionContext &txn, const ProjectedRow &low_key,
                           const ProjectedRow &high_key, std::vector<TupleSlot> *value_list,
                           const uint32_t limit) final {
    TERRIER_ASSERT(value_list->empty(), "Result set should begin empty.");
    TERRIER_ASSERT(limit > 0, "Limit must be greater than 0.");

    // Build search keys
    KeyType index_low_key, index_high_key;
    index_low_key.SetFromProjectedRow(low_key, metadata_, metadata_.GetSchema().GetColumns().size());
    index_high_key.SetFromProjectedRow(high_key, metadata_, metadata_.GetSchema().GetColumns().size());

    uint64_t epoch = bplustree_->StartFunction();

    std::function<bool(TupleSlot)> predicate = [&](TupleSlot tuple) { return IsVisible(txn, tuple); };

    // FIXME(15-721 project2): perform a lookup of the underlying data structure of the key
    for (auto *leaf = bplustree_->FindMaxLeafReadOnly(index_high_key); leaf != nullptr && value_list->size() < limit;
         leaf = leaf->left_) {
      if (!leaf->ScanRangeReverse(index_low_key, &index_high_key, value_list, predicate)) {
        break;
      }
    }

    while (value_list->size() > limit) {
      value_list->pop_back();
    }

    bplustree_->EndFunction(epoch);
  }
};

extern template class BPlusTreeIndex<CompactIntsKey<8>>;
extern template class BPlusTreeIndex<CompactIntsKey<16>>;
extern template class BPlusTreeIndex<CompactIntsKey<24>>;
extern template class BPlusTreeIndex<CompactIntsKey<32>>;

extern template class BPlusTreeIndex<GenericKey<64>>;
extern template class BPlusTreeIndex<GenericKey<128>>;
extern template class BPlusTreeIndex<GenericKey<256>>;

extern template class BPlusTreeIndex<uint64_t>;

}  // namespace terrier::storage::index
