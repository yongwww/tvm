## Rebase vs Merge

The backup unity branch is [unity-rebase-backup-2023-02-18](https://github.com/yongwww/tvm/commits/unity-rebase-backup-2023-02-18), git rebase on branch [unity-rebase-yongwww-2023-02-27-10am](https://github.com/yongwww/tvm/tree/unity-rebase-yongwww-2023-02-27-10am), and git cherry-pick on branch [unity-cherry-pick-yongwww-2023-02-27-10.5am](https://github.com/yongwww/tvm/tree/unity-cherry-pick-yongwww-2023-02-27-10.5am) on top of apache tvm/main with commit [bf589f3d1198d24997c468d094b3f6c22af6b42f](https://github.com/apache/tvm/commit/bf589f3d1198d24997c468d094b3f6c22af6b42f). There are 47 new commits were introduced in apache tvm main.&#x20;

### Rebase

```shell
git fetch upstream
git rebase upstream/main
```

Time: 26 minutes and 29 seconds. There is no conflict! Building and testing occupied most of the time.

### Merge

`git merge/pull commit-id` automatically generates message like "Merge commit 'commit-id' into \<Branch Name>", the commit message in the related uptream commit id is lost. I would like to maintain the same commit message as upstream, so I used `git cherry-pick` for this experiment instead.&#x20;

```shell
# colllect the commits to be cherry-picked
git checkout upstream/main
# 14bc5e45855f5a80b7c57a53d98bb6016c9bbf53 is the latest commit in both tvm main and unity 
git rev-list 14bc5e45855f5a80b7c57a53d98bb6016c9bbf53..HEAD > commits.txt
tac commits.txt > commits.sh
sed -i -e 's/^/git cherry-pick /' commits.sh
# cherry-pick
git checkout unity-cherry-pick-yongwww-2023-02-27-10.5am
sh commits.sh
rm commits.txt commits.sh
```
Time:  32 minutes and 08 seconds. Spent some extra time verifying the commit list manually.
