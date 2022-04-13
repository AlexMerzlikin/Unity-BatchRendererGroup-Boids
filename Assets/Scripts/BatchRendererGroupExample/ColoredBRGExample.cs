using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

namespace BatchRendererGroupExample
{
    public class ColoredBRGExample : MonoBehaviour
    {
        [SerializeField] private Mesh _mesh;
        [SerializeField] private Material _material;
        [SerializeField] private float _motionSpeed;
        [SerializeField] private float _motionAmplitude;
        [SerializeField] private Vector3 _motionDirection;
        [SerializeField] private uint _instancesCount = 1;
        [SerializeField] private float _radius;

        private BatchRendererGroup _brg;
        private GraphicsBuffer _instanceData;
        private BatchID _batchID;
        private BatchMeshID _meshID;
        private BatchMaterialID _materialID;
        private float _phase;
        private float3x4[] _objectToWorld;
        private float3x4[] _worldToObject;
        private uint _byteAddressWorldToObject;
        private uint _byteAddressObjectToWorld;

        private const int SizeOfFloat4 = sizeof(float) * 4;
        private const int SizeOfMatrix = SizeOfFloat4 * 4;
        private const int SizeOfPackedMatrix = SizeOfFloat4 * 3;
        private const int BytesPerInstance = SizeOfPackedMatrix * 2 + SizeOfFloat4;
        private const int Offset = 32;
        private const int ExtraBytes = SizeOfMatrix + Offset;

        private static int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
        {
            bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            var totalBytes = bytesPerInstance * numInstances + extraBytes;
            return totalBytes / sizeof(int);
        }

        private void Start()
        {
            _brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
            _meshID = _brg.RegisterMesh(_mesh);
            _materialID = _brg.RegisterMaterial(_material);

            var bufferCountForInstances = BufferCountForInstances(BytesPerInstance, (int) _instancesCount, ExtraBytes);
            _instanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
                bufferCountForInstances,
                sizeof(int));

            var zero = new Matrix4x4[] { Matrix4x4.zero };

            var matrices = new float4x4[_instancesCount];
            for (var i = 0; i < matrices.Length; i++)
            {
                matrices[i] = Matrix4x4.Translate(Random.onUnitSphere * _radius);
            }

            _objectToWorld = new float3x4[_instancesCount];
            for (var i = 0; i < _instancesCount; i++)
            {
                _objectToWorld[i] = new float3x4(
                    matrices[i].c0.x, matrices[i].c1.x, matrices[i].c2.x, matrices[i].c3.x,
                    matrices[i].c0.y, matrices[i].c1.y, matrices[i].c2.y, matrices[i].c3.y,
                    matrices[i].c0.z, matrices[i].c1.z, matrices[i].c2.z, matrices[i].c3.z
                );
            }

            _worldToObject = new float3x4[_instancesCount];
            for (var i = 0; i < _instancesCount; i++)
            {
                var inverse = math.inverse(matrices[i]);
                _worldToObject[i] = new float3x4(
                    inverse.c0.x, inverse.c1.x, inverse.c2.x, inverse.c3.x,
                    inverse.c0.y, inverse.c1.y, inverse.c2.y, inverse.c3.y,
                    inverse.c0.z, inverse.c1.z, inverse.c2.z, inverse.c3.z
                );
            }

            var colors = new Vector4[_instancesCount];
            for (var i = 0; i < _instancesCount; i++)
            {
                var color = new Color(
                    math.abs(_objectToWorld[i].c3.x) / _radius,
                    math.abs(_objectToWorld[i].c3.y) / _radius,
                    math.abs(_objectToWorld[i].c3.z) / _radius);
                colors[i] = new Vector4(color.r, color.g, color.b, color.a);
            }

            _byteAddressObjectToWorld = SizeOfPackedMatrix * 2;
            _byteAddressWorldToObject = _byteAddressObjectToWorld + SizeOfPackedMatrix * _instancesCount;
            var byteAddressColor = _byteAddressWorldToObject + SizeOfPackedMatrix * _instancesCount;

            _instanceData.SetData(zero, 0, 0, 1);
            _instanceData.SetData(_objectToWorld, 0, (int) (_byteAddressObjectToWorld / SizeOfPackedMatrix),
                _objectToWorld.Length);
            _instanceData.SetData(_worldToObject, 0, (int) (_byteAddressWorldToObject / SizeOfPackedMatrix),
                _worldToObject.Length);
            _instanceData.SetData(colors, 0, (int) (byteAddressColor / SizeOfFloat4), colors.Length);

            var metadata = new NativeArray<MetadataValue>(3, Allocator.Temp)
            {
                [0] = new MetadataValue
                {
                    NameID = Shader.PropertyToID("unity_ObjectToWorld"),
                    Value = 0x80000000 | _byteAddressObjectToWorld,
                },
                [1] = new MetadataValue
                {
                    NameID = Shader.PropertyToID("unity_WorldToObject"),
                    Value = 0x80000000 | _byteAddressWorldToObject,
                },
                [2] = new MetadataValue
                {
                    NameID = Shader.PropertyToID("_BaseColor"),
                    Value = 0x80000000 | byteAddressColor,
                }
            };

            _batchID = _brg.AddBatch(metadata, _instanceData.bufferHandle);
        }

        private void Update()
        {
            _phase += Time.fixedDeltaTime * _motionSpeed;
            var translation = _motionDirection * _motionAmplitude;
            var pos = translation * Mathf.Cos(_phase);
            UpdatePositions(pos);
        }

        private void UpdatePositions(Vector3 pos)
        {
            for (var i = 0; i < _instancesCount; i++)
            {
                _objectToWorld[i] = new float3x4(
                    new float3(_objectToWorld[i].c0.x, _objectToWorld[i].c0.y, _objectToWorld[i].c0.z),
                    new float3(_objectToWorld[i].c1.x, _objectToWorld[i].c1.y, _objectToWorld[i].c1.z),
                    new float3(_objectToWorld[i].c2.x, _objectToWorld[i].c2.y, _objectToWorld[i].c2.z),
                    new float3(_objectToWorld[i].c3.x + pos.x, _objectToWorld[i].c3.y + pos.y,
                        _objectToWorld[i].c3.z + pos.z));

                _worldToObject[i] = new float3x4(
                    new float3(_worldToObject[i].c0.x, _worldToObject[i].c0.y, _worldToObject[i].c0.z),
                    new float3(_worldToObject[i].c1.x, _worldToObject[i].c1.y, _worldToObject[i].c1.z),
                    new float3(_worldToObject[i].c2.x, _worldToObject[i].c2.y, _worldToObject[i].c2.z),
                    new float3(_worldToObject[i].c3.x - pos.x, _worldToObject[i].c3.y - pos.y,
                        _worldToObject[i].c3.z - pos.z));
            }

            _instanceData.SetData(_objectToWorld, 0, (int) (_byteAddressObjectToWorld / SizeOfPackedMatrix),
                _objectToWorld.Length);
            _instanceData.SetData(_worldToObject, 0, (int) (_byteAddressWorldToObject / SizeOfPackedMatrix),
                _worldToObject.Length);
        }

        private void OnDisable()
        {
            _instanceData.Dispose();
            _brg.Dispose();
        }

        private unsafe JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            var alignment = UnsafeUtility.AlignOf<long>();
            var drawCommands = (BatchCullingOutputDrawCommands*) cullingOutput.drawCommands.GetUnsafePtr();

            drawCommands->drawCommands = (BatchDrawCommand*) UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<BatchDrawCommand>(),
                alignment, Allocator.TempJob);
            drawCommands->drawRanges =
                (BatchDrawRange*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(), alignment,
                    Allocator.TempJob);
            drawCommands->visibleInstances =
                (int*) UnsafeUtility.Malloc(_instancesCount * sizeof(int), alignment, Allocator.TempJob);
            drawCommands->drawCommandPickingInstanceIDs = null;

            drawCommands->drawCommandCount = 1;
            drawCommands->drawRangeCount = 1;
            drawCommands->visibleInstanceCount = (int) _instancesCount;
            drawCommands->instanceSortingPositions = null;
            drawCommands->instanceSortingPositionFloatCount = 0;

            drawCommands->drawCommands[0].visibleOffset = 0;
            drawCommands->drawCommands[0].visibleCount = _instancesCount;
            drawCommands->drawCommands[0].batchID = _batchID;
            drawCommands->drawCommands[0].materialID = _materialID;
            drawCommands->drawCommands[0].meshID = _meshID;
            drawCommands->drawCommands[0].submeshIndex = 0;
            drawCommands->drawCommands[0].splitVisibilityMask = 0xff;
            drawCommands->drawCommands[0].flags = 0;
            drawCommands->drawCommands[0].sortingPosition = 0;

            drawCommands->drawRanges[0].drawCommandsBegin = 0;
            drawCommands->drawRanges[0].drawCommandsCount = 1;
            drawCommands->drawRanges[0].filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffff, };

            for (var i = 0; i < _instancesCount; ++i)
            {
                drawCommands->visibleInstances[i] = i;
            }

            return new JobHandle();
        }
    }
}