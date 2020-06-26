# LeetCode

 just code it

- 二叉树

  * 1) [计算从上到下任意起始和结尾节点的和为某一定值的路径个数 20200623](https://leetcode-cn.com/problems/path-sum-iii/  "don't stop")

    1. **找到最简单的子问题求解**
       * 终止条件
       * 分解成哪几部分
    2. **其他问题不考虑内在细节，只考虑整体逻辑**

    * [java题解思路](https://leetcode-cn.com/problems/path-sum-iii/solution/437lu-jing-zong-he-iii-di-gui-fang-shi-by-ming-zhi/)

  * 2) [判断一棵二叉树是否是对称二叉树  20200624 (使用递归解决了, 迭代如何实现?)](https://leetcode-cn.com/problems/symmetric-tree/)

    * 用两个指针镜像遍历, 判断是否相同
    
  * 3) [二叉树的直径: 任意两个结点路径长度中的最大值 20200625](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

    * 转化为求左右子树的深度和的最大值;
    * 求子树深度的终止条件是 if (node == null) == > return 0;
    * 所以递归方法的 deep(TreeNode node) 的返回值 =  左右子树中最大的深度 + 1(自身节点)
    * 使用 deep(node) 的方式 遍历整棵树, 在此过程中, 使用一个变量 ans 更新记录每次遍历后的最长路径所需要的经过的节点个数； 每次遍历后 ans = max(ans, left_deep + right_deep + 1)
    * 最后返回最长路径 = 最长路径经过的节点个数 - 1

- 链表

  - 1) [编写一个程序，找到两个单链表相交的起始节点 20200624](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)
    * [若相交，链表A： a+c, 链表B : b+c.   a+c+b+c = b+c+a+c 。则会在公共处c起点相遇。若不相交，a +b = b+a 。因此相遇处是NULL](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/tu-jie-xiang-jiao-lian-biao-by-user7208t/)

- 栈

  - 1) [设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈  20200624](https://leetcode-cn.com/problems/min-stack/)
    - [用一个额外的栈 stack_min 来降序保存最小的值, 保证栈顶一定是当前栈中最小的值](https://leetcode-cn.com/problems/min-stack/solution/min-stack-fu-zhu-stackfa-by-jin407891080/)

- 动态规划

  - 1) [给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。(亦可以用分治) 20200625](https://leetcode-cn.com/problems/maximum-subarray/)

    - 每次遍历的时候, 更新当前 sum 的最大值:  if cur<=0 ==> sum;  else  sum += cur;
    - ans = max(ans, sum) : 在每次遍历后使用 max() 方法 更新最优解

    > 分治思想: 
    >
    > 将纠结空间分成左半部分, 右半部分, 和包含中间 mid, mid+1的部分这3 块来求解.
    >
    > 最后取这三部分中的最大值为最终的结果, 使用递归的方式来实现.
    >
    > ![img](https://pic.leetcode-cn.com/a0f0a42149f9cebccb3ea4d8d1901d3d4ce934abd249149e2e6dbe84f17e14c2-01.png)
    
  - 2) [爬楼梯 20200626](https://leetcode-cn.com/problems/climbing-stairs/submissions/)

    - 斐波那契数列, 使用递归或者滚动数组的思想

      ![](https://assets.leetcode-cn.com/solution-static/70/70_fig1.gif)

