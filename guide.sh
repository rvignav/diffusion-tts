#interactive
cd /lustre/fsw/portfolios/nvr/users/mmardani/testtime_scaling
sh interactive_job.sh
cd /code/testtime_scaling
bash torchrun_script.sh

git add .
git commit -m "update"
git pull
#git push

rm -rf *.npy* *autog*