using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using URandom = UnityEngine.Random;

namespace ThousandAnt.Boids
{
    internal unsafe class PinnedMatrixArray : IDisposable
    {
        internal Matrix4x4[] Src; // Our source buffer for reading
        internal Matrix4x4[] Dst; // Our double buffer for writing

        // Using pinned pointers
        internal float4x4* SrcPtr;
        internal float4x4* DstPtr;

        internal int Size { get; private set; }

        private GCHandle _srcHandle;
        private GCHandle _dstHandle;

        internal PinnedMatrixArray(int size)
        {
            Src = new Matrix4x4[size];
            GCHandle.Alloc(Src, GCHandleType.Pinned);
            fixed (Matrix4x4* ptr = Src)
            {
                SrcPtr = (float4x4*) ptr;
            }

            Dst = new Matrix4x4[size];
            GCHandle.Alloc(Dst, GCHandleType.Pinned);

            fixed (Matrix4x4* ptr = Dst)
            {
                DstPtr = (float4x4*) ptr;
            }

            Size = size;
        }

        public void Dispose()
        {
            if (_srcHandle.IsAllocated)
            {
                _srcHandle.Free();
            }

            if (_dstHandle.IsAllocated)
            {
                _dstHandle.Free();
            }
        }
    }

    public unsafe class InstancedBoidsRunner : Runner
    {
        public Mesh Mesh;
        public Material Material;
        public ShadowCastingMode Mode;
        public bool ReceiveShadows;
        public Color Initial;
        public Color Final;

        private MaterialPropertyBlock tempBlock;
        private PinnedMatrixArray matrices;
        private NativeArray<float> noiseOffsets;
        private float3* centerFlock;
        private JobHandle boidsHandle;
        private Vector4[] colors;
        private static readonly int ColorProperty = Shader.PropertyToID("_Color");

        private void Start()
        {
            tempBlock = new MaterialPropertyBlock();
            matrices = new PinnedMatrixArray(Size);
            noiseOffsets = new NativeArray<float>(Size, Allocator.Persistent);
            colors = new Vector4[Size];

            for (int i = 0; i < Size; i++)
            {
                var pos = transform.position + URandom.insideUnitSphere * Radius;
                var rotation = Quaternion.Slerp(transform.rotation, URandom.rotation, 0.3f);
                noiseOffsets[i] = URandom.value * 10f;
                matrices.Src[i] = Matrix4x4.TRS(pos, rotation, Vector3.one);

                colors[i] = new Color(
                    URandom.Range(Initial.r, Final.r),
                    URandom.Range(Initial.b, Final.b),
                    URandom.Range(Initial.g, Final.g),
                    URandom.Range(Initial.a, Final.a));
            }

            tempBlock.SetVectorArray(ColorProperty, colors);

            centerFlock = (float3*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<float3>(),
                UnsafeUtility.AlignOf<float3>(),
                Allocator.Persistent);

            UnsafeUtility.MemSet(centerFlock, 0, UnsafeUtility.SizeOf<float3>());
        }

        private void OnDisable()
        {
            // Like the GameObjectBoidsRunner - complete all jobs before disabling
            boidsHandle.Complete();

            // Free this memory
            if (noiseOffsets.IsCreated)
            {
                noiseOffsets.Dispose();
            }

            if (centerFlock != null)
            {
                UnsafeUtility.Free(centerFlock, Allocator.Persistent);
                centerFlock = null;
            }
        }

        private void Update()
        {
            // Complete all jobs at the start of the frame.
            boidsHandle.Complete();

            // Set up the transform so that we have cinemachine to look at
            transform.position = *centerFlock;

            for (int i = 0; i < Mesh.subMeshCount; i++)
            {
                // Draw all elements, because we use a pinned array, the pointer is
                // representative of the array.
                Graphics.DrawMeshInstanced(
                    Mesh,
                    i,
                    Material,
                    matrices.Src, // Matrices.Src is an array (Matrix4x4[])
                    matrices.Src.Length,
                    tempBlock,
                    Mode,
                    ReceiveShadows,
                    0,
                    null);
            }

            var avgCenterJob = new BoidsPointerOnly.AverageCenterJob
            {
                Matrices = matrices.SrcPtr,
                Center = centerFlock,
                Size = matrices.Size
            }.Schedule();

            var boidJob = new BoidsPointerOnly.BatchedBoidJob
            {
                Weights = Weights,
                Goal = Destination.position,
                NoiseOffsets = noiseOffsets,
                Time = Time.time,
                DeltaTime = Time.deltaTime,
                MaxDist = SeparationDistance,
                Speed = MaxSpeed,
                RotationSpeed = RotationSpeed,
                Size = matrices.Size,
                Src = matrices.SrcPtr,
                Dst = matrices.DstPtr,
            }.Schedule(matrices.Size, 32);

            var combinedJob = JobHandle.CombineDependencies(boidJob, avgCenterJob);

            boidsHandle = new BoidsPointerOnly.CopyMatrixJob
            {
                Dst = matrices.SrcPtr,
                Src = matrices.DstPtr
            }.Schedule(matrices.Size, 32, combinedJob);
        }
    }
}