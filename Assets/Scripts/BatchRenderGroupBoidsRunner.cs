using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

namespace ThousandAnt.Boids
{
    public unsafe class BatchRenderGroupBoidsRunner : Runner
    {
        public Mesh Mesh;
        public Material Material;
        public Color Initial;
        public Color Final;

        private MaterialPropertyBlock tempBlock;
        private PinnedMatrixArray matrices;
        private NativeArray<float> noiseOffsets;
        private float3* centerFlock;
        private JobHandle boidsHandle;
        private Vector4[] colors;
        private static readonly int ColorProperty = Shader.PropertyToID("_Color");


        [SerializeField] private float m_motionSpeed = 3.0f;
        [SerializeField] private float m_motionAmplitude = 2.0f;
        [SerializeField] private float m_spacingFactor = 1.0f;

        private BatchRendererGroup m_BatchRendererGroup;
        private GraphicsBuffer m_GPUPersistentInstanceData;
        private NativeArray<Vector4> m_sysmemBuffer;
        private BatchID m_batchID;
        private BatchMaterialID m_materialID;
        private BatchMeshID m_meshID;
        private bool m_initialized;
        private float m_phase;


        private void Start()
        {
            InitBoids();
            InitBatchRendererGroup();
        }

        private void InitBatchRendererGroup()
        {
            m_BatchRendererGroup = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);

            // Bounds
            var bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
            m_BatchRendererGroup.SetGlobalBounds(bounds);

            // Register mesh and material
            if (Mesh)
            {
                m_meshID = m_BatchRendererGroup.RegisterMesh(Mesh);
            }

            if (Material)
            {
                m_materialID = m_BatchRendererGroup.RegisterMaterial(Material);
            }

            // Batch metadata buffer
            int objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
            int matrixPreviousMID = Shader.PropertyToID("unity_MatrixPreviousM");
            int worldToObjectID = Shader.PropertyToID("unity_WorldToObject");
            int colorID = Shader.PropertyToID("_BaseColor");

            // Generate a grid of objects...
            int bigDataBufferVector4Count =
                4 + Size * (3 * 3 + 1); // 4xfloat4 zero + per instance = { 3x mat4x3, 1x float4 color }
            m_sysmemBuffer = new NativeArray<Vector4>(bigDataBufferVector4Count, Allocator.Persistent);
            m_GPUPersistentInstanceData =
                new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) bigDataBufferVector4Count * 16 / 4, 4);

            // 64 bytes of zeroes, so loads from address 0 return zeroes. This is a BatchRendererGroup convention.
            const int positionOffset = 4;
            m_sysmemBuffer[0] = new Vector4(0, 0, 0, 0);
            m_sysmemBuffer[1] = new Vector4(0, 0, 0, 0);
            m_sysmemBuffer[2] = new Vector4(0, 0, 0, 0);
            m_sysmemBuffer[3] = new Vector4(0, 0, 0, 0);

            // Matrices
            UpdatePositions();

            // Colors
            int colorOffset = positionOffset + Size * 3 * 3;
            for (int i = 0; i < Size; i++)
            {
                Color col = Color.HSVToRGB((i / (float) Size) % 1.0f, 1.0f, 1.0f);

                // write colors right after the 4x3 matrices
                m_sysmemBuffer[colorOffset + i] = new Vector4(col.r, col.g, col.b, 1.0f);
            }

            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer);

            var batchMetadata =
                new NativeArray<MetadataValue>(4, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
                {
                    [0] = CreateMetadataValue(objectToWorldID, 64, true),
                    [1] = CreateMetadataValue(matrixPreviousMID, 64 + Size * UnsafeUtility.SizeOf<Vector4>() * 3,
                        true),
                    [2] = CreateMetadataValue(worldToObjectID,
                        64 + Size * UnsafeUtility.SizeOf<Vector4>() * 3 * 2,
                        true),
                    [3] = CreateMetadataValue(colorID, 64 + Size * UnsafeUtility.SizeOf<Vector4>() * 3 * 3, true)
                };
            // matrices
            // previous matrices
            // inverse matrices
            // colors

            // Register batch
            m_batchID = m_BatchRendererGroup.AddBatch(batchMetadata, m_GPUPersistentInstanceData.bufferHandle);

            m_initialized = true;
        }

        private void InitBoids()
        {
            tempBlock = new MaterialPropertyBlock();
            matrices = new PinnedMatrixArray(Size);
            noiseOffsets = new NativeArray<float>(Size, Allocator.Persistent);
            colors = new Vector4[Size];

            for (int i = 0; i < Size; i++)
            {
                var pos = transform.position + Random.insideUnitSphere * Radius;
                var rotation = Quaternion.Slerp(transform.rotation, Random.rotation, 0.3f);
                noiseOffsets[i] = Random.value * 10f;
                matrices.Src[i] = Matrix4x4.TRS(pos, rotation, Vector3.one);

                colors[i] = new Color(
                    Random.Range(Initial.r, Final.r),
                    Random.Range(Initial.b, Final.b),
                    Random.Range(Initial.g, Final.g),
                    Random.Range(Initial.a, Final.a));
            }

            tempBlock.SetVectorArray(ColorProperty, colors);

            centerFlock = (float3*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<float3>(),
                UnsafeUtility.AlignOf<float3>(),
                Allocator.Persistent);

            UnsafeUtility.MemSet(centerFlock, 0, UnsafeUtility.SizeOf<float3>());
        }


        private void Update()
        {
            // Complete all jobs at the start of the frame.
            boidsHandle.Complete();

            // Set up the transform so that we have cinemachine to look at
            transform.position = *centerFlock;

            UpdatePositions();
            // upload the full buffer
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer);
            // for (int i = 0; i < Mesh.subMeshCount; i++)
            // {
            //     // Draw all elements, because we use a pinned array, the pointer is
            //     // representative of the array.
            //     Graphics.DrawMeshInstanced(
            //         Mesh,
            //         i,
            //         Material,
            //         matrices.Src, // Matrices.Src is an array (Matrix4x4[])
            //         matrices.Src.Length,
            //         tempBlock);
            // }

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

        private void UpdatePositions()
        {
            int positionOffset = 4;
            int itemCountOffset = Size * Size * 3; // 3xfloat4 per matrix

            for (int z = 0; z < Size; z++)
            {
                {
                    int i = z;

                    /*
                     *  mat4x3 packed like this:
                     *
                            float4x4(
                                    p1.x, p1.w, p2.z, p3.y,
                                    p1.y, p2.x, p2.w, p3.z,
                                    p1.z, p2.y, p3.x, p3.w,
                                    0.0, 0.0, 0.0, 1.0
                                );
                    */

                    // update previous matrix with previous frame current matrix
                    // m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset] =
                    //     m_sysmemBuffer[positionOffset + i * 3 + 0];
                    // m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset] =
                    //     m_sysmemBuffer[positionOffset + i * 3 + 1];
                    // m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset] =
                    //     m_sysmemBuffer[positionOffset + i * 3 + 2];

                    // m_sysmemBuffer[positionOffset + i * 3 + 0] = new Vector4(1, 0, 0, 0);
                    // m_sysmemBuffer[positionOffset + i * 3 + 1] = new Vector4(1, 0, 0, 0);
                    // m_sysmemBuffer[positionOffset + i * 3 + 2] = new Vector4(1, px + pos.x, pos.y, pz + pos.z);
                    //
                    // // compute the new inverse matrix
                    // m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset * 2] = new Vector4(1, 0, 0, 0);
                    // m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset * 2] = new Vector4(1, 0, 0, 0);
                    // m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset * 2] = new Vector4(1, -(px + pos.x), -pos.y, -(pz + pos.z));
                    
                    
                    // compute the new current frame matrix
                    m_sysmemBuffer[positionOffset + i * 3 + 0] = new Vector4(1, 0,0, 0);
                    m_sysmemBuffer[positionOffset + i * 3 + 1] = new Vector4(1, 0,0, 0);
                    m_sysmemBuffer[positionOffset + i * 3 + 2] = new Vector4(1, matrices.Src[z].m21,matrices.Src[z].m22, matrices.Src[z].m23);

                    // var a = Matrix4x4.Inverse(matrices.Src[z]);


                    // compute the new inverse matrix
                    m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset] = new Vector4(1, 0, 0, 0);
                    m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset] = new Vector4(1, 0, 0, 0);
                    m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset] = new Vector4(1, -matrices.Src[z].m21,-matrices.Src[z].m22, -matrices.Src[z].m23);
                    
                    // compute the new inverse matrix
                    // m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset] = new Vector4(a.m00, a.m01, a.m02, a.m03);
                    // m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset] = new Vector4(a.m10, a.m11, a.m12, a.m13);
                    // m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset] = new Vector4(a.m20, a.m21, a.m22, a.m23);
                }
            }
        }


        public unsafe JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            if (!m_initialized)
            {
                return new JobHandle();
            }

            BatchCullingOutputDrawCommands drawCommands = new BatchCullingOutputDrawCommands();

            drawCommands.drawRangeCount = 1;
            drawCommands.drawRanges = Malloc<BatchDrawRange>(1);
            drawCommands.drawRanges[0] = new BatchDrawRange
            {
                drawCommandsBegin = 0,
                drawCommandsCount = 1,
                filterSettings = new BatchFilterSettings
                {
                    renderingLayerMask = 1,
                    layer = 0,
                    shadowCastingMode = ShadowCastingMode.On,
                    receiveShadows = true,
                    staticShadowCaster = false,
                    allDepthSorted = false
                }
            };

            drawCommands.visibleInstances = Malloc<int>(Size);
            int n = 0;
            for (int r = 0; r < Size; r++)
            {
                drawCommands.visibleInstances[n++] = r;
            }

            drawCommands.visibleInstanceCount = n;

            drawCommands.drawCommandCount = 1;
            drawCommands.drawCommands = Malloc<BatchDrawCommand>(1);
            drawCommands.drawCommands[0] = new BatchDrawCommand
            {
                visibleOffset = 0,
                visibleCount = (uint) n,
                batchID = m_batchID,
                materialID = m_materialID,
                meshID = m_meshID,
                submeshIndex = 0,
                splitVisibilityMask = 0xff,
                sortingPosition = 0
            };


            drawCommands.instanceSortingPositions = null;
            drawCommands.instanceSortingPositionFloatCount = 0;

            cullingOutput.drawCommands[0] = drawCommands;
            return new JobHandle();
        }

        private static unsafe T* Malloc<T>(int count) where T : unmanaged
        {
            return (T*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<T>() * count,
                UnsafeUtility.AlignOf<T>(),
                Allocator.TempJob);
        }

        private static MetadataValue CreateMetadataValue(int nameID, int gpuAddress, bool isOverridden)
        {
            const uint kIsOverriddenBit = 0x80000000;
            return new MetadataValue
            {
                NameID = nameID,
                Value = (uint) gpuAddress | (isOverridden ? kIsOverriddenBit : 0),
            };
        }

        private void OnDisable()
        {
            DisposeBoids();
            DisposeBatchRendererGroup();
        }

        private void DisposeBoids()
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

        private void DisposeBatchRendererGroup()
        {
            if (!m_initialized)
            {
                return;
            }

            m_BatchRendererGroup.RemoveBatch(m_batchID);
            if (Material)
            {
                m_BatchRendererGroup.UnregisterMaterial(m_materialID);
            }

            if (Mesh)
            {
                m_BatchRendererGroup.UnregisterMesh(m_meshID);
            }

            m_BatchRendererGroup.Dispose();
            m_GPUPersistentInstanceData.Dispose();
            m_sysmemBuffer.Dispose();
        }
    }
}