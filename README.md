### **just_code_it**



[树](#树)

[数组](#数组)

[栈](#栈)

[链表](#链表)

[动态规划](动态规划)

[回溯](#回溯)

[递归](#递归)

- ## 树

  1. #####   [计算从上到下任意起始和结尾节点的和为某一定值的路径个数 20200623](https://leetcode-cn.com/problems/path-sum-iii/  "don't stop")

       1. **找到最简单的子问题求解**
          - 终止条件
          - 分解成哪几部分
     2. **其他问题不考虑内在细节，只考虑整体逻辑**
       3. [java题解思路](https://leetcode-cn.com/problems/path-sum-iii/solution/437lu-jing-zong-he-iii-di-gui-fang-shi-by-ming-zhi/)

  2. #####  [判断一棵二叉树是否是对称二叉树  20200624 (使用递归解决了, 迭代如何实现?)](https://leetcode-cn.com/problems/symmetric-tree/)

     - 用两个指针镜像遍历, 判断是否相同

  3. #####  [二叉树的直径: 任意两个结点路径长度中的最大值 20200625](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

       - 转化为求左右子树的深度和的最大值;
       - 求子树深度的终止条件是 if (node == null) == > return 0;
       - 所以递归方法的 deep(TreeNode node) 的返回值 =  左右子树中最大的深度 + 1(自身节点)
       - 使用 deep(node) 的方式 遍历整棵树, 在此过程中, 使用一个变量 ans 更新记录每次遍历后的最长路径所需要的经过的节点个数； 每次遍历后 ans = max(ans, left_deep + right_deep + 1)
       - 最后返回最长路径 = 最长路径经过的节点个数 - 1

  4. #####  [22.括号生成 20200708](https://leetcode-cn.com/problems/generate-parentheses/)

     ![](https://pic.leetcode-cn.com/7ec04f84e936e95782aba26c4663c5fe7aaf94a2a80986a97d81574467b0c513-LeetCode%20%E7%AC%AC%2022%20%E9%A2%98%EF%BC%9A%E2%80%9C%E6%8B%AC%E5%8F%B7%E7%94%9F%E5%87%BA%E2%80%9D%E9%A2%98%E8%A7%A3%E9%85%8D%E5%9B%BE.png)

       1. 当前左右括号都有大于 00 个可以使用的时候，才产生分支；
       2. 产生左分支的时候，只看当前是否还有左括号可以使用；
       3. 产生右分支的时候，还受到左分支的限制，右边剩余可以使用的括号数量一定得在严格大于左边剩余的数量的时候，才可以产生分支；
       4. 在左边和右边剩余的括号数都等于 00 的时候结算。

  5. #####  [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

     实现一个 Trie (前缀树)，包含 `insert`, `search`, 和 `startsWith` 这三个操作。

  ```java
  class TrieNode {
          private boolean isEnd;
          TrieNode[] next;
  
          public TrieNode() {
              this.isEnd = false;
              this.next = new TrieNode[26];
          }
      }
  ```

  6. #####  [96. 不同的二叉搜索树 20200729](https://leetcode-cn.com/problems/unique-binary-search-trees/)

       - 找到数学公式即可

  7. #####  [105. 从前序与中序遍历序列构造二叉树 20200729](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

       - 递归
       - 利用栈遍历

  8. ##### [236. 二叉树的最近公共祖先 20200801](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

- ## 数组

  1. ##### [最短无序连续子数组 20200703](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

       - 自己想的方法是, 从左到右遍历找到逆序中的最小值 min , 并更新当前遇到的最大值,以此来找到右边界; 再从右边届向左遍历, 找出左边界.
     - 看到两个比较巧妙的方法:
         1. 向右遍历, 用 left = nums.length-1 记录左边界; 记录当前的最大值max_previous; 当遇到cur_Num < max_previous 的时候, 就更新 left 的值 , 试着将所有比 cur_Num 大的数放到 左边界 left 右边去;  同时, 更新 right = cur_Num.index() .  这样 一次遍历就找出了左右边界.
       2. 使用一次for循环, 分别从两端相向而行来分别更新左右边界, 简洁而巧妙. 
  
```java
  class Solution {
    public int findUnsortedSubarray(int[] nums) {
          int len = nums.length;
          int max = nums[0];
          int min = nums[len-1];
          int l = 0, r = -1;
          for(int i=0;i<len;i++){
              if(max>nums[i]){
                  r = i;
              }else{
                  max = nums[i];
              }
              if(min<nums[len-i-1]){
                  l = len-i-1;
              }else{
                  min = nums[len-i-1];
              }
          }
          return r-l+1;
      }
  }
  ```
  
  2. #####  [238. 除自身以外数组的乘积 20200711](https://leetcode-cn.com/problems/product-of-array-except-self/)
  
     - 分别从左到右和从右到左两趟遍历来计算.
  
```java
  class Solution {
    public int[] productExceptSelf(int[] nums) {
          int len = nums.length;
          int[] ans = new int[len];
          ans[0] = 1;
          for(int i=1; i<len; i++) { // 从左到右依次计算得到每个点左边的乘积
              ans[i] = ans[i-1] * nums[i-1];
          }
          int right_multi = nums[len-1];
          for(int i=len-2; i>=0; i--) { // 从右到左, 依次计算得到右边的乘积并整合
              ans[i] *= right_multi;
              right_multi *= nums[i];
          }
          return ans;
      }
  }
  ```
  
  3. #####  [48. 旋转图像 20200715](https://leetcode-cn.com/problems/rotate-image/)
  
  - > 给定一个 *n* × *n* 的二维矩阵表示一个图像。
      >
    > 将图像顺时针旋转 90 度。
  
        - 方法一: 先转置矩阵，然后翻转每一行
        - 方法二: 剥洋葱式层层翻转, 注: 外层循环多一层

  ![](https://pic.leetcode-cn.com/12605efb60d2efc64e6ecfcf6562a98a49acb3ce696b0c1ad3da46ab8977fa16-48_angles.png)

  4. #####  [64. 最小路径和 20200730](https://leetcode-cn.com/problems/minimum-path-sum/)

       - 将二维dp用一维表示, 在另一个维度里再依次更新

  5. #####  [406. 根据身高重建队列 20200731](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

     - 先进行二维排序, 再依次按照第二个下标插入新的数组中

  6. ##### [287. 寻找重复数 20200731](https://leetcode-cn.com/problems/find-the-duplicate-number/)

     - 由于空间复杂度要求为O(1), 不能使用哈希表冲突判断法
   - 可以使用二分法, 记录左右边界 left 和 right, 每次计算中间值mid, 和在mid左边的数字的数量cnt.  如果 cnt>mid ,   表示重复的数字在cnt左半边, 否则重复的数字在cnt右半边. 当 left == right 的时候 表示找到了重复的数字.
  
- ## 链表

  1. #####  [编写一个程序，找到两个单链表相交的起始节点 20200624](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)
  
       - [若相交，链表A： a+c, 链表B : b+c.   a+c+b+c = b+c+a+c 。则会在公共处c起点相遇。若不相交，a +b = b+a 。因此相遇处是NULL](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/tu-jie-xiang-jiao-lian-biao-by-user7208t/)
  
  2. #####  [判断链表是否有环 20200629](https://leetcode-cn.com/problems/linked-list-cycle/) 

     - 使用快慢指针, 快的步长为2, 慢的步长为1. 如果存在环则快慢指针定会相遇; 否则快指针会先到达链尾

  3. #####  [判断链表是否回文 20200701](https://leetcode-cn.com/problems/palindrome-linked-list/submissions/)

       - 法1: 快慢指针+栈, 在慢指针到达中间的时候开始判断是否回文
       - 法2: 利用快慢指针快速找到中间节点的同时, 将前半部分的链表指针翻转, 再从中间向两端遍历判断是否相同以构成回文. 最后将链表指针顺序恢复. **(空间利用率更低, 速度更快)**
  
4. #####  [206. 反转链表 20200710](https://leetcode-cn.com/problems/reverse-linked-list/)
  
      - <details><summary>递归方法: 先处理后边的, 再处理当前的</summary><pre>
            class Solution {
            private ListNode res;
            private ListNode temp;
            public ListNode reverseList(ListNode head) {
                if(head==null || head.next==null)
                    return head;
                ListNode ptr = head;
                dfs(ptr);
                return res;
            }
            private void dfs(ListNode ptr) {
                if(ptr.next != null ){
                    dfs(ptr.next);
                    temp.next = new ListNode(ptr.val);
                    temp = temp.next;
                }
                else{
                    res = new ListNode(ptr.val);
                    temp = res;
                }
            }
        }
        </details>
  
      - <details><summary>遍历方法: 头插法</summary><pre>
        class Solution {
            public ListNode reverseList(ListNode head) {
                if(head == null)
                    return head;
                ListNode ans = null;
                while(head != null){
                    ListNode h = head.next;
                    head.next = ans;
                    ans = head;
                    head = h;
                }
                return ans;
            }
        }
  
  5. #####  [328. 奇偶链表 20200711](https://leetcode-cn.com/problems/odd-even-linked-list/)
  
      - **示例:**
  
      ```
      输入: 2->1->3->5->6->4->7->NULL 
      输出: 2->3->6->7->1->5->4->NULL
      ```
  
      - <details><summary>用两个指针交替拆线将原链表分成两个分别保存奇偶序号节点的链表, 最终完成拼接</summary><pre>
        class Solution {
            public ListNode oddEvenList(ListNode head) {
                if(head == null || head.next == null)
                    return head;
                ListNode odd_tail = head;
                ListNode even_head = head.next;
                ListNode even_tail = even_head;
                ListNode t;
                while(even_tail != null && even_tail.next != null) {
                    odd_tail.next = even_tail.next;
                    odd_tail = odd_tail.next;
                    even_tail.next = odd_tail.next;
                    even_tail = even_tail.next;
                }
                odd_tail.next = even_head;
                return head;
            }
        }
        最终会出现两种情况:
        	1) odd_tail -> even_tail(null) 
        	2) odd_tail -> even_tail -> null
        在 odd_tail.next = even_head; 后都能完成拼接操作.
  
  6. #####  [148. 排序链表 20200730](https://leetcode-cn.com/problems/sort-list/)
  
      - 归并(用快慢指针找中心点)
      - 快排序(分为less和more两个子链表, 递归求解)
  
  7. 

---



- ## 栈

  1. #####  [设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈  20200624](https://leetcode-cn.com/problems/min-stack/)

       - [用一个额外的栈 stack_min 来降序保存最小的值, 保证栈顶一定是当前栈中最小的值](https://leetcode-cn.com/problems/min-stack/solution/min-stack-fu-zhu-stackfa-by-jin407891080/)

  2. #####  [有效的括号 20200702](https://leetcode-cn.com/problems/valid-parentheses/)

     ```python
     for c in string: # 遍历字符串
         if c is 左括号:
             stack.push(c)  # 如果是左括号就入栈
         else:
             if stack.isEmpty():  # 若是右括号且栈为空, 即栈中没有能与之匹配的左括号
                 return False
            	if 左右括号匹配:
                 stack.pop()  # 左右括号匹配, 将栈中左括号弹出
             else:
                 return False  # 栈虽不为空, 但是栈顶左括号与当前右括号不匹配, 不合法
             
     return stack.isEmpty()  # 仅当遍历结束后栈为空时, 字符串合法
     
     ```

  3. 



---



- ## 动态规划

  1. #####  [给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。(亦可以用分治) 20200625](https://leetcode-cn.com/problems/maximum-subarray/)

       - 每次遍历的时候, 更新当前 sum 的最大值:  if cur<=0 ==> sum;  else  sum += cur;
       - ans = max(ans, sum) : 在每次遍历后使用 max() 方法 更新最优解

    > 分治思想: 
    >
    > 将纠结空间分成左半部分, 右半部分, 和包含中间 mid, mid+1的部分这3 块来求解.
    >
    > 最后取这三部分中的最大值为最终的结果, 使用递归的方式来实现.
    >
    > ![img](https://pic.leetcode-cn.com/a0f0a42149f9cebccb3ea4d8d1901d3d4ce934abd249149e2e6dbe84f17e14c2-01.png)
  
  2. #####  [爬楼梯 20200626](https://leetcode-cn.com/problems/climbing-stairs/submissions/)

       - 斐波那契数列, 使用递归或者滚动数组的思想

  ![](https://assets.leetcode-cn.com/solution-static/70/70_fig1.gif)

  3. #####  [打家劫舍(一排房屋中有不同的钱财, 不能连续抢劫, 求最优能抢到的金额 )  20200630](https://leetcode-cn.com/problems/house-robber/submissions/)

       - 用变量 S<sub>n</sub>保存抢前n家所能获得的最大的金额, 用 M<sub>n</sub>表示第n家的金额
     - S<sub>n</sub> = max (S<sub>n-2</sub> + M<sub>n</sub>  ,   S<sub>n-1</sub>)
     - 为了减少空间利用率, 可以用滚动数组的方式.

  4. 

---



- ## 回溯

  1. #####  [78.子集 20200706](https://leetcode-cn.com/problems/subsets/)

    - 
       遍历思想
      <details><summary>展开代码</summary><pre>
      //python3
      class Solution:
          def subsets(self, nums: List[int]) -> List[List[int]]:
              res = [[]]
              for num in nums:
                  res += [[num] + arr for arr in res]
              return res;
      -----------------------------------------------------------------
      //java
      class Solution {
        public List<List<Integer>> subsets(int[] nums) {
          List<List<Integer>> res = new ArrayList();
          res.add(new ArrayList<Integer>());
          for (int num:nums){
              List<List<Integer>> tempArr = new ArrayList();
              for (List<Integer> subArr:res){
                  tempArr.add(new ArrayList<Integer>(subArr){{add(num);}});
              }
              for (List<Integer> arr:tempArr)
                  res.add(arr);
          }
          return res;
        }
      }
      </pre>    
      </details>
      
    - 递归思想

        <details><summary>展开代码</summary><pre>
        class Solution:
            def subsets(self, nums: List[int]) -> List[List[int]]:
                res = []
                n = len(nums)
                def helper(i, temp):
                    res.append(temp)
                    for j in range(i, n):
                        helper(j+1, temp + [nums[j]])
                helper(0, [])
                return res;
      
    - java 回溯

      <details><summary>展开代码</summary><pre>
      class Solution {
        public List<List<Integer>> subsets(int[] nums) {
          List<List<Integer>> res = new ArrayList();
          backtrack(0, nums, res, new ArrayList<Integer>());
          return res;
        }
        public void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> temp) {
          res.add(new ArrayList<Integer>(temp));
          for (int j=i; j<>nums.length; j++) {
              temp.add(nums[j]);
              backtrack(j+1, nums, res, temp);
              temp.remove(temp.size()-1);
          }
        }
      }

  2. #####  [全排列 20200706](https://leetcode-cn.com/problems/permutations/)

      - 利用树形结构 + 回溯 

      ![](https://pic.leetcode-cn.com/0bf18f9b86a2542d1f6aa8db6cc45475fce5aa329a07ca02a9357c2ead81eec1-image.png)

      - 使用 path 记录深度优先遍历的路径, used[]  记录节点是否在path中
      - 终止条件: 当path长度等于数组长度的时候, 将 path添加到 res 数组中
      - 在此过程中如果节点不在path中, 就将其加入并设置uesd 为true, 然后dfs递归. 在递归结束后恢复递归前的状态, 即将最新加入的节点删除并置used为false, 此即回溯的含义.

      ```java
        for(int i=0; i<nums.length; i++) {
                  if(!used[i]){
                        used[i] = true;
                        path.add(nums[i]);
                        dfs(used, nums, res, path, depth+1);
                        used[i] = false;
                        path.remove(path.size()-1);
                    }
                }
      ```

  3. #####  [39. 组合总和 20200714](https://leetcode-cn.com/problems/combination-sum/)

    - 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

      candidates 中的数字可以无限制重复被选取。

    - 遍历candidates, 记录path, 记录当前遍历的index, 在未满足条件的情况下使用当前index作为起始点继续寻找, 使得该点能够被重复添加到path中; 在寻找的过程中, 没加入一个点, 就在target中减去这个值, 当target == 0 的时候, 就将path中的点组成的list加入到res中.

  4. 

- ## 递归

  1. #####  [338.比特位计数 20200709](https://leetcode-cn.com/problems/counting-bits/)

       - 一个数乘以2后其就是左移一位, 1 的个数不会改变, 一个偶数a, a+1 的 1 的个数是 res[a] + 1

     <details><summary>展开 java 递归代码</summary><pre>
         class Solution {
         private int[] res;
         public int[] countBits(int num) {
             res = new int[num + 1];
             if (num == 0) {
                 return res;
             }
             res[1] = 1;
             helper(num, 1, 1);
             return res;
         }
         private void helper(int num, int i, int count) {
             i = i << 1; // i *= 2
             if (i <= num) {
                 res[i] = count; // 左移1的个数不改变
                 helper(num, i, count);
             }
             i += 1; // 此时 i 必定为奇数
             if (i <= num) {
                 res[i] = count + 1; 此奇数的1的个数必定是前一个偶数的1的个数+1
                 helper(num, i, count + 1);
             }
         }
     }
     </details>

  2. #####  [94. 二叉树的中序遍历 20200710](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

       - 使用递归

  ```java
  class Solution {
      public List<Integer> inorderTraversal(TreeNode root) {
          List<Integer> res = new ArrayList();
          dfs(res, root);
          return res;
      }
  
      private void dfs(List<Integer> res, TreeNode node) {
          if(node == null)
              return;
          dfs(res, node.left);
          res.add(node.val);
          dfs(res, node.right);
      }
  }
  ```

  3. #####  [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

       - 在递归中进行变换: 从最底层开始变换(先递归再变换)

  ```java
  public void flatten(TreeNode root) {
          if(root == null)
              return;
  		/*这两句可以调换顺序, 不影响结果, 区别是默认优先从左子树还是右字数进行变换*/
          flatten(root.left); // (1)
          flatten(root.right); // (2) 
  
          // 暂存右子树, 左边移到右边, 左边置为空
          TreeNode temp = root.right;
          root.right = root.left;
          root.left = null;
  
          // 移到当前右子树的最右下端, 将旧的右子树接到其下
          while(root.right != null)
              root = root.right;
          root.right = temp;
      }
  ```

  4. 

[回到顶部](#just_code_it)
