using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;
using URandom = UnityEngine.Random;

namespace ThousandAnt.Boids {

    // Mark the class as unsafe because we want to make use of pointers.
    public unsafe class GameObjectsBoidsRunner : Runner {

        public Transform FlockMember;
        public bool UseSingleThread;

        private NativeArray<float> noiseOffsets;
        private NativeArray<float4x4> srcMatrices;
        private NativeArray<float4x4> dstMatrices;
        private Transform[] transforms;
        private TransformAccessArray transformAccessArray;
        private JobHandle boidsHandle;
        private float3* center;

        private void Start() {
            // We spawn n GameObjects that we will manipulate. We store the positions and their associated noise offsets.
            // The nosie offsts are useful for providing a unique sense of movement per element.
            transforms   = new Transform[Size];
            srcMatrices  = new NativeArray<float4x4>(transforms.Length, Allocator.Persistent);
            dstMatrices  = new NativeArray<float4x4>(transforms.Length, Allocator.Persistent);
            noiseOffsets = new NativeArray<float>(transforms.Length, Allocator.Persistent);

            for (int i = 0; i < Size; i++) {
                var pos         = transform.position + URandom.insideUnitSphere * Radius;
                var rotation    = Quaternion.Slerp(transform.rotation, URandom.rotation, 0.3f);
                transforms[i]   = GameObject.Instantiate(FlockMember, pos, rotation) as Transform;
                srcMatrices[i]  = transforms[i].localToWorldMatrix;
                noiseOffsets[i] = URandom.value * 10f;
            }

            // Create the transform access array with a cache of Transforms.
            transformAccessArray = new TransformAccessArray(transforms);

            // To pass from a Job struct back to our MonoBehaviour, we need to use a pointer. In newer packages there is
            // NativeReference<T> which serves the same purpose as a pointer. This allows us to write the position
            // back to our pointer so we can read it later in the main thread to use.
            center = (float3*)UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<float3>(),
                UnsafeUtility.AlignOf<float3>(),
                Allocator.Persistent);

            // Set the pointer to the float3 to be the default value, or float3.zero.
            UnsafeUtility.MemSet(center, default, UnsafeUtility.SizeOf<float3>());
        }

        private void OnDisable() {
            // Before this component is disabled, make sure that all the jobs are completed.
            boidsHandle.Complete();

            // Then we dispose all the NativeArrays we allocate.
            if (srcMatrices.IsCreated) {
                srcMatrices.Dispose();
            }

            if (dstMatrices.IsCreated) {
                dstMatrices.Dispose();
            }

            if (noiseOffsets.IsCreated) {
                noiseOffsets.Dispose();
            }

            if (transformAccessArray.isCreated) {
                transformAccessArray.Dispose();
            }

            if (center != null) {
                UnsafeUtility.Free(center, Allocator.Persistent);
                center = null;
            }
        }

        private unsafe void Update() {
            // At the start of the frame, we ensure that all the jobs scheduled are completed.
            boidsHandle.Complete();

            // Write the contents from the pointer back to our position.
            transform.position = *center;

            // Copy the contents from the NativeArray to our TransformAccess
            var copyTransformJob = new CopyTransformJob {
                Src = srcMatrices
            }.Schedule(transformAccessArray);

            // Use a separate single thread to calculate the average center of the flock.
            var avgCenterJob = new AverageCenterJob {
                Matrices = srcMatrices,
                Center   = center,
            }.Schedule();

            JobHandle boidJob;

            // Compute boid - selectively use a multithreaded job or a single threaded job.
            if (!UseSingleThread) {
                boidJob           = new BatchedBoidJob {
                    Weights       = Weights,
                    Goal          = Destination.position,
                    NoiseOffsets  = noiseOffsets,
                    Time          = Time.time,
                    DeltaTime     = Time.deltaTime,
                    MaxDist       = SeparationDistance,
                    Speed         = MaxSpeed,
                    RotationSpeed = RotationSpeed,
                    Size          = srcMatrices.Length,
                    Src           = srcMatrices,
                    Dst           = dstMatrices
                }.Schedule(transforms.Length, 32);
            } else {
                boidJob = new BoidJob {
                    Weights       = Weights,
                    Goal          = Destination.position,
                    NoiseOffsets  = noiseOffsets,
                    Time          = Time.time,
                    DeltaTime     = Time.deltaTime,
                    MaxDist       = SeparationDistance,
                    Speed         = MaxSpeed,
                    RotationSpeed = RotationSpeed,
                    Size          = srcMatrices.Length,
                    Src           = srcMatrices,
                    Dst           = dstMatrices
                }.Schedule();
            }

            // Combine all jobs to a single dependency, so we can pass this single dependency to the
            // CopyMatrixJob. The CopyMatrixJob needs to wait until all jobs are done so we can avoid
            // concurrency issues.
            var combinedJob = JobHandle.CombineDependencies(avgCenterJob, boidJob, copyTransformJob);

            boidsHandle = new CopyMatrixJob {
                Dst = srcMatrices,
                Src = dstMatrices
            }.Schedule(srcMatrices.Length, 32, combinedJob);
        }
    }
}
