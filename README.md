# LeetCode

 just code it

- ## 二叉树

  * ### 1)  [计算从上到下任意起始和结尾节点的和为某一定值的路径个数 20200623](https://leetcode-cn.com/problems/path-sum-iii/  "don't stop")

    1. **找到最简单的子问题求解**
       * 终止条件
       * 分解成哪几部分
    2. **其他问题不考虑内在细节，只考虑整体逻辑**

    * [java题解思路](https://leetcode-cn.com/problems/path-sum-iii/solution/437lu-jing-zong-he-iii-di-gui-fang-shi-by-ming-zhi/)

  * ### 2) [判断一棵二叉树是否是对称二叉树  20200624 (使用递归解决了, 迭代如何实现?)](https://leetcode-cn.com/problems/symmetric-tree/)

    * 用两个指针镜像遍历, 判断是否相同
    
  * ### 3) [二叉树的直径: 任意两个结点路径长度中的最大值 20200625](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

    * 转化为求左右子树的深度和的最大值;
    * 求子树深度的终止条件是 if (node == null) == > return 0;
    * 所以递归方法的 deep(TreeNode node) 的返回值 =  左右子树中最大的深度 + 1(自身节点)
    * 使用 deep(node) 的方式 遍历整棵树, 在此过程中, 使用一个变量 ans 更新记录每次遍历后的最长路径所需要的经过的节点个数； 每次遍历后 ans = max(ans, left_deep + right_deep + 1)
    * 最后返回最长路径 = 最长路径经过的节点个数 - 1



- ## 数组

  - ### 1) [最短无序连续子数组 20200703](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

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

         

- ## 链表

  - ### 1) [编写一个程序，找到两个单链表相交的起始节点 20200624](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)
    
    * [若相交，链表A： a+c, 链表B : b+c.   a+c+b+c = b+c+a+c 。则会在公共处c起点相遇。若不相交，a +b = b+a 。因此相遇处是NULL](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/tu-jie-xiang-jiao-lian-biao-by-user7208t/)
    
  - ### 2) [判断链表是否有环 20200629](https://leetcode-cn.com/problems/linked-list-cycle/) 

    - 使用快慢指针, 快的步长为2, 慢的步长为1. 如果存在环则快慢指针定会相遇; 否则快指针会先到达链尾

  - ### 3) [判断链表是否回文 20200701](https://leetcode-cn.com/problems/palindrome-linked-list/submissions/)

    - 法1: 快慢指针+栈, 在慢指针到达中间的时候开始判断是否回文

    - 法2: 利用快慢指针快速找到中间节点的同时, 将前半部分的链表指针翻转, 再从中间向两端遍历判断是否相同以构成回文. 最后将链表指针顺序恢复. **(空间利用率更低, 速度更快)**



---



- ## 栈

  - ### 1) [设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈  20200624](https://leetcode-cn.com/problems/min-stack/)
    
    - [用一个额外的栈 stack_min 来降序保存最小的值, 保证栈顶一定是当前栈中最小的值](https://leetcode-cn.com/problems/min-stack/solution/min-stack-fu-zhu-stackfa-by-jin407891080/)
    
  - ### 2) [有效的括号 20200702](https://leetcode-cn.com/problems/valid-parentheses/)
  
    - ```python
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
  
      



---



- ## 动态规划

  - ### 1) [给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。(亦可以用分治) 20200625](https://leetcode-cn.com/problems/maximum-subarray/)

    - 每次遍历的时候, 更新当前 sum 的最大值:  if cur<=0 ==> sum;  else  sum += cur;
    - ans = max(ans, sum) : 在每次遍历后使用 max() 方法 更新最优解

    > 分治思想: 
    >
    > 将纠结空间分成左半部分, 右半部分, 和包含中间 mid, mid+1的部分这3 块来求解.
    >
    > 最后取这三部分中的最大值为最终的结果, 使用递归的方式来实现.
    >
    > ![img](https://pic.leetcode-cn.com/a0f0a42149f9cebccb3ea4d8d1901d3d4ce934abd249149e2e6dbe84f17e14c2-01.png)
    
  - ### 2) [爬楼梯 20200626](https://leetcode-cn.com/problems/climbing-stairs/submissions/)

    - 斐波那契数列, 使用递归或者滚动数组的思想

      ![](https://assets.leetcode-cn.com/solution-static/70/70_fig1.gif)

  - ### 3) [打家劫舍(一排房屋中有不同的钱财, 不能连续抢劫, 求最优能抢到的金额 )  20200630](https://leetcode-cn.com/problems/house-robber/submissions/)

    - 用变量 S<sub>n</sub>保存抢前n家所能获得的最大的金额, 用 M<sub>n</sub>表示第n家的金额

    - S<sub>n</sub> = max (S<sub>n-2</sub> + M<sub>n</sub>  ,   S<sub>n-1</sub>)

    - 为了减少空间利用率, 可以用滚动数组的方式.



---



- ## 回溯

  - ### 1) [78.子集 20200706](https://leetcode-cn.com/problems/subsets/)

    - 
       遍历思想
      ```
      //python3
      class Solution:
          def subsets(self, nums: List[int]) -> List[List[int]]:
              res = [[]]
              for num in nums:
                  res += [[num] + arr for arr in res]
              return res;
      
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
      ```

    - 递归思想

      ```
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
      ```

      

    - java 回溯

      ```
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
      ```

  - ### [全排列 20200706](https://leetcode-cn.com/problems/permutations/)

    - 利用树形结构 + 回溯 

      ![](https://pic.leetcode-cn.com/0bf18f9b86a2542d1f6aa8db6cc45475fce5aa329a07ca02a9357c2ead81eec1-image.png)

    - 使用 path 记录深度优先遍历的路径, used[]  记录节点是否在path中

    - 终止条件: 当path长度等于数组长度的时候, 将 path添加到 res 数组中

    - 在此过程中如果节点不再path中, 就将其加入并设置uesd 为true, 然后dfs递归. 在递归结束后恢复递归前的状态, 即将最新加入的节点删除并置used为false, 此即回溯的含义.

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

      

