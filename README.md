# MDR

Using-GPU：A100-40G*8

Used Training data: Link：https://pan.baidu.com/s/1tF8CSWhVbwevfAPaipvRYQ?pwd=7nui  password：7nui 

Run run.sh to run our step-by step code, You shoule replace the data path to  your own path. 
you can download GME-7B model and dino-v2 as backbone.
the pipeline is :
1. first using cv_tools dino to recognize the visual keypoints, here you get a visual-keypoints results.
2. train five different parameters expert to vote for both two tasks, the five experts total is a unified model.
3. merge the experts' voting results and  the visual keypoints results to get
The five experts and the dino as a whole can be viewed as a unified model. They are useful for both tasks and do not need to be trained specifically for the task. 
You can download Several dataset by M2KR-train-dataset. You need to transfer the json file to CSV file and merge the two results so that you can submit it to the leaderboard. 

You can run a single Expert as a simple version of our system by run run2.sh
