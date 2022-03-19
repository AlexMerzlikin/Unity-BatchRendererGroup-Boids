using System;
using ThousandAnt.Boids;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

namespace BatchRendererGroupExample
{
    public unsafe class BatchRenderGroupBoidsRunner : Runner
    {
        [SerializeField] private Mesh _mesh;
        [SerializeField] private Material _material;

        private PinnedMatrixArray _matrices;
        private NativeArray<float> _noiseOffsets;
        private float3* _centerFlock;
        private JobHandle _boidsHandle;

        private BatchRendererGroup _batchRendererGroup;
        private GraphicsBuffer _gpuPersistentInstanceData;
        private NativeArray<Vector4> _dataBuffer;
        private BatchID _batchID;
        private BatchMaterialID _materialID;
        private BatchMeshID _meshID;
        private bool _initialized;

        private void Start()
        {
            InitBoids();
            InitBatchRendererGroup();
        }

        private void InitBatchRendererGroup()
        {
            _batchRendererGroup = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);

            // Bounds
            var bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
            _batchRendererGroup.SetGlobalBounds(bounds);

            // Register mesh and material
            if (_mesh)
            {
                _meshID = _batchRendererGroup.RegisterMesh(_mesh);
            }

            if (_material)
            {
                _materialID = _batchRendererGroup.RegisterMaterial(_material);
            }

            // Batch metadata buffer
            var objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
            var worldToObjectID = Shader.PropertyToID("unity_WorldToObject");

            // Generate a grid of objects...
            var bigDataBufferVector4Count = 4 + Size * (2 * 3); // 4xfloat4 zero + per instance = { 3x mat4x3, 1x float4 color }
            _dataBuffer = new NativeArray<Vector4>(bigDataBufferVector4Count, Allocator.Persistent);
            _gpuPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, bigDataBufferVector4Count * 16 / 4, 4);

            // 64 bytes of zeroes, so loads from address 0 return zeroes. This is a BatchRendererGroup convention.
            const int positionOffset = 4 * 4 * sizeof(float);
            _dataBuffer[0] = new Vector4(0, 0, 0, 0);
            _dataBuffer[1] = new Vector4(0, 0, 0, 0);
            _dataBuffer[2] = new Vector4(0, 0, 0, 0);
            _dataBuffer[3] = new Vector4(0, 0, 0, 0);

            // Matrices
            UpdatePositions();
            _gpuPersistentInstanceData.SetData(_dataBuffer);
            var batchMetadata =
                new NativeArray<MetadataValue>(2, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
                {
                    [0] = CreateMetadataValue(objectToWorldID, positionOffset, true),
                    [1] = CreateMetadataValue(worldToObjectID,
                        positionOffset + Size * UnsafeUtility.SizeOf<Vector4>() * 2, true),
                };

            // Register batch
            _batchID = _batchRendererGroup.AddBatch(batchMetadata, _gpuPersistentInstanceData.bufferHandle);


            _initialized = true;
        }

        private void InitBoids()
        {
            _matrices = new PinnedMatrixArray(Size);
            _noiseOffsets = new NativeArray<float>(Size, Allocator.Persistent);

            for (var i = 0; i < Size; i++)
            {
                var currentTransform = transform;
                var pos = currentTransform.position + Random.insideUnitSphere * Radius;
                var rotation = Quaternion.Slerp(currentTransform.rotation, Random.rotation, 0.3f);
                _noiseOffsets[i] = Random.value * 10f;
                _matrices.Src[i] = Matrix4x4.TRS(pos, rotation, Vector3.one);
            }

            _centerFlock = (float3*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<float3>(),
                UnsafeUtility.AlignOf<float3>(),
                Allocator.Persistent);

            UnsafeUtility.MemSet(_centerFlock, 0, UnsafeUtility.SizeOf<float3>());
        }


        private void Update()
        {
            // Complete all jobs at the start of the frame.
            _boidsHandle.Complete();
            // Set up the transform so that we have cinemachine to look at
            transform.position = *_centerFlock;

            UpdatePositions();
            // upload the full buffer
            _gpuPersistentInstanceData.SetData(_dataBuffer);

            var avgCenterJob = new BoidsPointerOnly.AverageCenterJob
            {
                Matrices = _matrices.SrcPtr,
                Center = _centerFlock,
                Size = _matrices.Size
            }.Schedule();

            var boidsJob = new BoidsPointerOnly.BatchedBoidJob
            {
                Weights = Weights,
                Goal = Destination.position,
                NoiseOffsets = _noiseOffsets,
                Time = Time.time,
                DeltaTime = Time.deltaTime,
                MaxDist = SeparationDistance,
                Speed = MaxSpeed,
                RotationSpeed = RotationSpeed,
                Size = _matrices.Size,
                Src = _matrices.SrcPtr,
                Dst = _matrices.DstPtr,
            }.Schedule(_matrices.Size, 32);

            var combinedJob = JobHandle.CombineDependencies(boidsJob, avgCenterJob);

            _boidsHandle = new BoidsPointerOnly.CopyMatrixJob
            {
                Dst = _matrices.SrcPtr,
                Src = _matrices.DstPtr
            }.Schedule(_matrices.Size, 32, combinedJob);
        }

        private void UpdatePositions()
        {
            const int positionOffset = 4;
            var itemCountOffset = 3 * Size; // 3xfloat4 per matrix

            for (var i = 0; i < Size; i++)
            {
                {
                    // compute the new current frame matrix
                    _dataBuffer[positionOffset + i * 3 + 0] = new Vector4(_matrices.Src[i].m00, _matrices.Src[i].m10, _matrices.Src[i].m20, _matrices.Src[i].m01);
                    _dataBuffer[positionOffset + i * 3 + 1] = new Vector4(_matrices.Src[i].m11, _matrices.Src[i].m21, _matrices.Src[i].m02, _matrices.Src[i].m12);
                    _dataBuffer[positionOffset + i * 3 + 2] = new Vector4(_matrices.Src[i].m22, _matrices.Src[i].m03, _matrices.Src[i].m13, _matrices.Src[i].m23);

                    // compute the new inverse matrix
                    var inverse = Matrix4x4.Inverse(_matrices.Src[i]);
                    _dataBuffer[positionOffset + i * 3 + 0 + itemCountOffset] = new Vector4(inverse.m00, inverse.m10, inverse.m20, inverse.m01);
                    _dataBuffer[positionOffset + i * 3 + 1 + itemCountOffset] = new Vector4(inverse.m11, inverse.m21, inverse.m02, inverse.m12);
                    _dataBuffer[positionOffset + i * 3 + 2 + itemCountOffset] = new Vector4(inverse.m22, inverse.m03, inverse.m13, inverse.m23);
                }
            }
        }


        private JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            if (!_initialized)
            {
                return new JobHandle();
            }

            var drawCommands = new BatchCullingOutputDrawCommands();

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
            var n = 0;
            for (var r = 0; r < Size; r++)
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
                batchID = _batchID,
                materialID = _materialID,
                meshID = _meshID,
                submeshIndex = 0,
                splitVisibilityMask = 0xff,
                sortingPosition = 0
            };


            drawCommands.instanceSortingPositions = null;
            drawCommands.instanceSortingPositionFloatCount = 0;

            cullingOutput.drawCommands[0] = drawCommands;
            return new JobHandle();
        }

        private static T* Malloc<T>(int count) where T : unmanaged
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
            _boidsHandle.Complete();

            // Free this memory
            if (_noiseOffsets.IsCreated)
            {
                _noiseOffsets.Dispose();
            }

            if (_centerFlock != null)
            {
                UnsafeUtility.Free(_centerFlock, Allocator.Persistent);
                _centerFlock = null;
            }
        }

        private void DisposeBatchRendererGroup()
        {
            if (!_initialized)
            {
                return;
            }

            _batchRendererGroup.RemoveBatch(_batchID);
            if (_material)
            {
                _batchRendererGroup.UnregisterMaterial(_materialID);
            }

            if (_mesh)
            {
                _batchRendererGroup.UnregisterMesh(_meshID);
            }

            _batchRendererGroup.Dispose();
            _gpuPersistentInstanceData.Dispose();
            _dataBuffer.Dispose();
        }
    }
}