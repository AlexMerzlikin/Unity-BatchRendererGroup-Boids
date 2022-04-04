# Unity-BatchRendererGroup-Boids

A simple example how to use the new BatchRendererGroup API to render boids made for my blog at https://gamedev.center/trying-out-new-unity-api-batchrenderergroup/. The post to be published soon.

For calculating boids behaviour I used [ta-boids](https://github.com/ThousandAnt/ta-boids).

However, for me this boids solution turned out to be a bit unstable, as even little performance dips or interaction with the editor breaks the __centerFlock_ pointer, leaves it as (NaN, Nan, NaN) and boids stop working both for my BRG variant, as well as GameObject and Instanced variants provided in the boids repo. 
So in the editor I could only test boids with a small amount of objects up to 2k. When I have found out this issue, it was already too late and I was too lazy to find a new boids lib and redo BRG variant again.
Anyway it suits my goal to test BRG compared to GameObjects and Instancing. 

# References
https://forum.unity.com/threads/new-batchrenderergroup-api-for-2022-1.1230669/

Examples by Unity: https://github.com/Unity-Technologies/Graphics/tree/master/TestProjects/BatchRendererGroup_URP/Assets/SampleScenes
