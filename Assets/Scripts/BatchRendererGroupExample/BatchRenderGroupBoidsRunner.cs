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
            var bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
            _batchRendererGroup.SetGlobalBounds(bounds);

            if (_mesh)
            {
                _meshID = _batchRendererGroup.RegisterMesh(_mesh);
            }

            if (_material)
            {
                _materialID = _batchRendererGroup.RegisterMaterial(_material);
            }

            var objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
            var worldToObjectID = Shader.PropertyToID("unity_WorldToObject");

            const int matrixSizeInFloats = 4;
            const int packedMatrixSizeInFloats = 3;
            const int matricesPerInstance = 2;
            // float4x4.zero + per instance data = { 2 * float3x4 }
            var bigDataBufferVector4Count = matrixSizeInFloats + Size * packedMatrixSizeInFloats * matricesPerInstance; 
            _dataBuffer = new NativeArray<Vector4>(bigDataBufferVector4Count, Allocator.Persistent);
            var bigDataBufferFloatCount = bigDataBufferVector4Count * 4;
            _gpuPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, bigDataBufferFloatCount, 4);

            // 64 bytes of zeroes, so loads from address 0 return zeroes. This is a BatchRendererGroup convention.
            _dataBuffer[0] = Vector4.zero;
            _dataBuffer[1] = Vector4.zero;
            _dataBuffer[2] = Vector4.zero;
            _dataBuffer[3] = Vector4.zero;

            _gpuPersistentInstanceData.SetData(_dataBuffer);
            var positionOffset = UnsafeUtility.SizeOf<Matrix4x4>();
            var inverseGpuAddress = positionOffset + Size * UnsafeUtility.SizeOf<float3x4>();
            var batchMetadata =
                new NativeArray<MetadataValue>(2, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
                {
                    [0] = CreateMetadataValue(objectToWorldID, positionOffset, true),
                    [1] = CreateMetadataValue(worldToObjectID, inverseGpuAddress, true),
                };

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
            _boidsHandle.Complete();
            transform.position = *_centerFlock;
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

            var copyJob = new CopyMatricesJob
            {
                DataBuffer = _dataBuffer,
                Size = _matrices.Size,
                Source = _matrices.SrcPtr
            }.Schedule(_matrices.Size, 32);
            
            var combinedJob = JobHandle.CombineDependencies(boidsJob, avgCenterJob, copyJob);

            _boidsHandle = new BoidsPointerOnly.CopyMatrixJob
            {
                Dst = _matrices.SrcPtr,
                Src = _matrices.DstPtr
            }.Schedule(_matrices.Size, 32, combinedJob);
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
            _boidsHandle.Complete();
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