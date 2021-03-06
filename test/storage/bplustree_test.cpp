
#include <limits>
#include <memory>

#include "portable_endian/portable_endian.h"
#include "storage/index/index_builder.h"
#include "test_util/test_harness.h"

namespace terrier::storage::index {

struct BPlusTreeTests : public TerrierTest {
  BPlusTree<uint64_t, uint64_t> *test_tree_;
  uint64_t num_inserts_ = 100000;
};

TEST_F(BPlusTreeTests, CompressTest1) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }

  uint64_t start_depth = test_tree_->GetDepth();

  uint64_t num_start_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_start_leaves++;

  for (uint64_t j = 0; j < num_inserts_; j++) {
    EXPECT_TRUE(test_tree_->Remove(j, j));
  }

  uint64_t del_depth = test_tree_->GetDepth();

  uint64_t num_end_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_end_leaves++;

  EXPECT_EQ(num_start_leaves, num_end_leaves);
  EXPECT_EQ(start_depth, del_depth);

  test_tree_->CompressTree();

  uint64_t compress_depth = test_tree_->GetDepth();

  uint64_t compressed_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    compressed_leaves++;

  EXPECT_EQ(compress_depth, start_depth);
  uint64_t estimated_leaves = num_inserts_ / (BPlusTree<uint64_t, uint64_t>::LEAF_SIZE / 2) /
                              (BPlusTree<uint64_t, uint64_t>::BRANCH_FACTOR / 2);
  EXPECT_TRUE(estimated_leaves == 0 || compressed_leaves >= (estimated_leaves - 1));
  EXPECT_TRUE(estimated_leaves == std::numeric_limits<uint64_t>::max() || compressed_leaves <= (estimated_leaves + 1));
}

TEST_F(BPlusTreeTests, CompressTest2) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }

  uint64_t start_depth = test_tree_->GetDepth();

  uint64_t num_start_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_start_leaves++;

  for (uint64_t j = 0; j < num_inserts_; j += 2) {
    EXPECT_TRUE(test_tree_->Remove(j, j));
  }

  uint64_t del_depth = test_tree_->GetDepth();

  uint64_t num_end_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_end_leaves++;

  EXPECT_EQ(num_start_leaves, num_end_leaves);
  EXPECT_EQ(start_depth, del_depth);

  test_tree_->CompressTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 1; i < num_inserts_; i += 2) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
}

TEST_F(BPlusTreeTests, CompressTest3) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE + 1; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }

  uint64_t start_depth = test_tree_->GetDepth();

  uint64_t num_start_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_start_leaves++;

  for (uint64_t j = 0; j < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE + 1; j += 2) {
    EXPECT_TRUE(test_tree_->Remove(j, j));
  }

  uint64_t del_depth = test_tree_->GetDepth();

  uint64_t num_end_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_end_leaves++;

  EXPECT_EQ(num_start_leaves, num_end_leaves);
  EXPECT_EQ(start_depth, del_depth);

  test_tree_->CompressTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 1; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE + 1; i += 2) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
}

TEST_F(BPlusTreeTests, CompressTest4) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE / 2; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }

  uint64_t start_depth = test_tree_->GetDepth();

  uint64_t num_start_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_start_leaves++;

  for (uint64_t j = 0; j < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE / 2; j += 2) {
    EXPECT_TRUE(test_tree_->Remove(j, j));
  }

  uint64_t del_depth = test_tree_->GetDepth();

  uint64_t num_end_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr;
       cur_node = cur_node->right_)
    num_end_leaves++;

  EXPECT_EQ(num_start_leaves, num_end_leaves);
  EXPECT_EQ(start_depth, del_depth);

  test_tree_->CompressTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 1; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE / 2; i += 2) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
}

TEST_F(BPlusTreeTests, RepeatInserts) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  test_tree_->CheckTree();
  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = true;
    EXPECT_FALSE(test_tree_->Insert(i, i, &rand, [](uint64_t i) { return true; }));
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();
}

TEST_F(BPlusTreeTests, SimpleInsert) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  test_tree_->CheckTree();
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();
}

TEST_F(BPlusTreeTests, SplitInsert) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  test_tree_->CheckTree();
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE + 1; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE + 1; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();
}

TEST_F(BPlusTreeTests, SimpleDelete) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  test_tree_->CheckTree();
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i += 2) {
    EXPECT_TRUE(test_tree_->Remove(i, i));
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < BPlusTree<uint64_t, uint64_t>::LEAF_SIZE; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    if (i % 2 == 0) {
      EXPECT_EQ(results.size(), 0);
    } else {
      EXPECT_EQ(results.size(), 1);
      EXPECT_EQ(results[0], i);
    }
    results.clear();
  }
  test_tree_->CheckTree();
}

TEST_F(BPlusTreeTests, FullDelete) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  test_tree_->CheckTree();
  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert(i, i, &rand));
  }
  test_tree_->CheckTree();

  std::vector<uint64_t> results;
  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0], i);
    results.clear();
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i += 2) {
    EXPECT_TRUE(test_tree_->Remove(i, i));
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    if (i % 2 == 0) {
      EXPECT_EQ(results.size(), 0);
    } else {
      EXPECT_EQ(results.size(), 1);
      EXPECT_EQ(results[0], i);
    }
    results.clear();
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    if (i % 2 == 0) {
      EXPECT_FALSE(test_tree_->Remove(i, i));
    } else {
      EXPECT_TRUE(test_tree_->Remove(i, i));
    }
  }
  test_tree_->CheckTree();

  for (uint64_t i = 0; i < num_inserts_; i++) {
    test_tree_->ScanKey(i, &results, [](uint64_t val) { return true; });
    EXPECT_EQ(results.size(), 0);
    results.clear();
  }
  test_tree_->CheckTree();
}

}  // namespace terrier::storage::index
