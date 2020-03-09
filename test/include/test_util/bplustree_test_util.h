#pragma once
#include "gtest/gtest.h"
#include "test_util/random_test_util.h"
#include <storage/index/bplustree.h>

class BPlusTreeTestUtil;
namespace terrier {
/**
 * Normally we restrict the scope of util files to named directories/namespaces (storage, transaction, etc.) but the
 * BwTree is a unique case because it's third party code but we want to test it rigorously. This isn't a
 * third_party_test_util because directories/files with "third_party" in their name are treated differently by the CI
 * scripts.
 */

struct BPlusTreeTestUtil {
  BPlusTreeTestUtil() = delete;

  /*
   * class KeyComparator - Test whether BwTree supports context
   *                       sensitive key comparator
   *
   * If a context-sensitive KeyComparator object is being used
   * then it should follow rules like:
   *   1. There could be no default constructor
   *   2. There MUST be a copy constructor
   *   3. operator() must be const
   *
   */
  class KeyComparator {
   public:
    bool operator()(const int64_t k1, const int64_t k2) const { return k1 < k2; }

    explicit KeyComparator(int dummy UNUSED_ATTRIBUTE) {}

    KeyComparator() = delete;
  };

  /*
   * class KeyEqualityChecker - Tests context sensitive key equality
   *                            checker inside BwTree
   *
   * NOTE: This class is only used in KeyEqual() function, and is not
   * used as STL template argument, it is not necessary to provide
   * the object everytime a container is initialized
   */
  class KeyEqualityChecker {
   public:
    bool operator()(const int64_t k1, const int64_t k2) const { return k1 == k2; }

    explicit KeyEqualityChecker(int dummy UNUSED_ATTRIBUTE) {}

    KeyEqualityChecker() = delete;
  };

  using TreeType = storage::index::BPlusTree<int64_t, int64_t, BPlusTreeTestUtil::KeyComparator, BPlusTreeTestUtil::KeyEqualityChecker>;

  /**
   * Adapted from https://github.com/wangziqi2013/BwTree/blob/master/test/test_suite.cpp
   */
  static TreeType *GetEmptyTree() {
    auto *tree = new TreeType{BPlusTreeTestUtil::KeyComparator{1}, BPlusTreeTestUtil::KeyEqualityChecker{1}};
    return tree;
  }
};

}  // namespace terrier
