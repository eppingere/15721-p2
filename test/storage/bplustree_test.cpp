
#include <limits>
#include <memory>

#include "portable_endian/portable_endian.h"
#include "storage/index/index_builder.h"
#include "test_util/test_harness.h"

namespace terrier::storage::index {

struct BPlusTreeTests : public TerrierTest {
  BPlusTree<uint64_t, uint64_t> *test_tree_;
 public:
  uint64_t num_inserts_ = 12000000;
};



// NOLINTNEXTLINE
TEST_F(BPlusTreeTests, EmptyTest) { EXPECT_TRUE(true); }

TEST_F(BPlusTreeTests, CompressTest) {
  test_tree_ = new BPlusTree<uint64_t, uint64_t>();
  for (uint64_t i = 0; i < num_inserts_; i++) {
    bool rand = false;
    EXPECT_TRUE(test_tree_->Insert([](uint64_t) { return false; }, i, i, &rand));
  }

  uint64_t start_depth = test_tree_->GetDepth();

  uint64_t num_start_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr; cur_node = cur_node->right_)
    num_start_leaves ++;

  for (uint64_t j = 0; j < num_inserts_; j++) {
    EXPECT_TRUE(test_tree_->Remove(j,j));
  }

  uint64_t del_depth = test_tree_->GetDepth();

  uint64_t num_end_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr; cur_node = cur_node->right_)
    num_end_leaves ++;

  EXPECT_EQ(num_start_leaves, num_end_leaves);
  EXPECT_EQ(start_depth, del_depth);

  test_tree_->CompressTree();

  uint64_t compress_depth = test_tree_->GetDepth();

  uint64_t compressed_leaves = 0;
  for (BPlusTree<uint64_t, uint64_t>::LeafNode *cur_node = test_tree_->FindMinLeaf(); cur_node != nullptr; cur_node = cur_node->right_)
    compressed_leaves ++;

  EXPECT_EQ(compress_depth, start_depth);
  EXPECT_EQ(compressed_leaves, num_start_leaves / (BPlusTree<uint64_t, uint64_t >::LEAF_SIZE / 2));
}

}  // namespace terrier::storage::index
