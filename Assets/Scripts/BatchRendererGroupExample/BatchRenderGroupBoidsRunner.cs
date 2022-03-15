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
        [SerializeField] private Color _initial;
        [SerializeField] private Color _final;

        private MaterialPropertyBlock _tempBlock;
        private PinnedMatrixArray _matrices;
        private NativeArray<float> _noiseOffsets;
        private float3* _centerFlock;
        private JobHandle _boidsHandle;
        private Vector4[] _colors;
        private static readonly int ColorProperty = Shader.PropertyToID("_Color");

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
            // var matrixPreviousMID = Shader.PropertyToID("unity_MatrixPreviousM");
            var worldToObjectID = Shader.PropertyToID("unity_WorldToObject");
            var colorID = Shader.PropertyToID("_BaseColor");

            // Generate a grid of objects...
            var bigDataBufferVector4Count = 4 + Size * (2 * 3); // 4xfloat4 zero + per instance = { 3x mat4x3, 1x float4 color }
            _dataBuffer = new NativeArray<Vector4>(bigDataBufferVector4Count, Allocator.Persistent);
            _gpuPersistentInstanceData =
                new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) bigDataBufferVector4Count * 16 / 4, 4);

            // 64 bytes of zeroes, so loads from address 0 return zeroes. This is a BatchRendererGroup convention.
            const int positionOffset = 4;
            _dataBuffer[0] = new Vector4(0, 0, 0, 0);
            _dataBuffer[1] = new Vector4(0, 0, 0, 0);
            _dataBuffer[2] = new Vector4(0, 0, 0, 0);
            _dataBuffer[3] = new Vector4(0, 0, 0, 0);

            // Matrices
            try
            {
                UpdatePositions();
                // Colors
                // var colorOffset = positionOffset + Size * 3 * 3;
                // for (var i = 0; i < Size; i++)
                // {
                //     var col = Color.HSVToRGB((i / (float) Size) % 1.0f, 1.0f, 1.0f);
                //
                //     // write colors right after the 4x3 matrices
                //     _dataBuffer[colorOffset + i] = new Vector4(col.r, col.g, col.b, 1.0f);
                // }

                _gpuPersistentInstanceData.SetData(_dataBuffer);

                var batchMetadata =
                    new NativeArray<MetadataValue>(2, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
                    {
                        [0] = CreateMetadataValue(objectToWorldID, 64, true),
                        // [1] = CreateMetadataValue(matrixPreviousMID, 64 + Size * UnsafeUtility.SizeOf<Vector4>() * 3,
                            // true),
                        [1] = CreateMetadataValue(worldToObjectID,
                            64 + Size * UnsafeUtility.SizeOf<Vector4>() * 2,
                            true),
                        // [2] = CreateMetadataValue(colorID, 64 + Size * UnsafeUtility.SizeOf<Vector4>() * 3 * 2, true)
                    };

                // Register batch
                _batchID = _batchRendererGroup.AddBatch(batchMetadata, _gpuPersistentInstanceData.bufferHandle);

            }
            catch (Exception e)
            {
                Debug.Log($"{nameof(BatchRenderGroupBoidsRunner)}: {e}");
            }

            _initialized = true;
        }

        private void InitBoids()
        {
            _tempBlock = new MaterialPropertyBlock();
            _matrices = new PinnedMatrixArray(Size);
            _noiseOffsets = new NativeArray<float>(Size, Allocator.Persistent);
            _colors = new Vector4[Size];

            for (var i = 0; i < Size; i++)
            {
                var pos = transform.position + Random.insideUnitSphere * Radius;
                var rotation = Quaternion.Slerp(transform.rotation, Random.rotation, 0.3f);
                _noiseOffsets[i] = Random.value * 10f;
                _matrices.Src[i] = Matrix4x4.TRS(pos, rotation, Vector3.one);

                _colors[i] = new Color(
                    Random.Range(_initial.r, _final.r),
                    Random.Range(_initial.b, _final.b),
                    Random.Range(_initial.g, _final.g),
                    Random.Range(_initial.a, _final.a));
            }

            _tempBlock.SetVectorArray(ColorProperty, _colors);

            _centerFlock = (float3*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<float3>(),
                UnsafeUtility.AlignOf<float3>(),
                Allocator.Persistent);

            UnsafeUtility.MemSet(_centerFlock, 0, UnsafeUtility.SizeOf<float3>());
        }


        private void Update()
        {
            // Complete all jobs at the start of the frame.
            try
            {
                _boidsHandle.Complete();
                // Set up the transform so that we have cinemachine to look at
                transform.position = *_centerFlock;

                UpdatePositions();
                // upload the full buffer
                _gpuPersistentInstanceData.SetData(_dataBuffer);
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
                    Matrices = _matrices.SrcPtr,
                    Center = _centerFlock,
                    Size = _matrices.Size
                }.Schedule();

                var boidJob = new BoidsPointerOnly.BatchedBoidJob
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

                var combinedJob = JobHandle.CombineDependencies(boidJob, avgCenterJob);

                _boidsHandle = new BoidsPointerOnly.CopyMatrixJob
                {
                    Dst = _matrices.SrcPtr,
                    Src = _matrices.DstPtr
                }.Schedule(_matrices.Size, 32, combinedJob);
            }
            catch (Exception e)
            {
                Debug.Log($"{nameof(BatchRenderGroupBoidsRunner)}: {e}");
            }
        }

        private void UpdatePositions()
        {
            const int positionOffset = 4;
            var itemCountOffset = 6; // 3xfloat4 per matrix

            for (var i = 0; i < Size; i++)
            {
                {
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

                    // _dataBuffer[positionOffset + i * 3 + 0 + itemCountOffset] = _dataBuffer[positionOffset + i * 3 + 0];
                    // _dataBuffer[positionOffset + i * 3 + 1 + itemCountOffset] = _dataBuffer[positionOffset + i * 3 + 1];
                    // _dataBuffer[positionOffset + i * 3 + 2 + itemCountOffset] = _dataBuffer[positionOffset + i * 3 + 2];
                    //
                    // compute the new current frame matrix
                    _dataBuffer[positionOffset + i * 3 + 0] = new Vector4(1, 0,0, 0); 
                    _dataBuffer[positionOffset + i * 3 + 1] = new Vector4(1, 0,0, 0);
                    _dataBuffer[positionOffset + i * 3 + 2] = new Vector4(1, _matrices.Src[i].m03,_matrices.Src[i].m13, _matrices.Src[i].m23);

                    // compute the new inverse matrix
                    _dataBuffer[positionOffset + i * 3 + 0 + itemCountOffset] = new Vector4(1, 0, 0, 0);
                    _dataBuffer[positionOffset + i * 3 + 1 + itemCountOffset] = new Vector4(1, 0, 0, 0);
                    _dataBuffer[positionOffset + i * 3 + 2 + itemCountOffset] = new Vector4(1, -_matrices.Src[i].m03,-_matrices.Src[i].m13, -_matrices.Src[i].m23);
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