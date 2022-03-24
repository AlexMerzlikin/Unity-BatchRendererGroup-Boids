using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

// This example demonstrates how to write a very minimal BatchRendererGroup
// based custom renderer using the Universal Render Pipeline to help
// getting started with using BatchRendererGroup.
public class SimpleBRGExample : MonoBehaviour
{
    // Set this to a suitable Mesh via the Inspector, such as a Cube mesh
    [SerializeField] private Mesh _mesh;

    // Set this to a suitable Material via the Inspector, such as a default material that
    // uses Universal Render Pipeline/Lit
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

    // Some helper constants to make calculations later a bit more convenient.
    private const int SizeOfMatrix = sizeof(float) * 4 * 4;
    private const int SizeOfPackedMatrix = sizeof(float) * 4 * 3;
    private const int BytesPerInstance = SizeOfPackedMatrix * 2;
    private const int Offset = 32;
    private const int ExtraBytes = SizeOfMatrix + Offset;

    // Raw buffers are allocated in ints, define an utility method to compute the required
    // amount of ints for our data.
    private static int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
    {
        // Round byte counts to int multiples
        bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        var totalBytes = bytesPerInstance * numInstances + extraBytes;
        return totalBytes / sizeof(int);
    }

    // During initialization, we will allocate all required objects, and set up our custom instance data.
    private void Start()
    {
        // Create the BatchRendererGroup and register assets
        _brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
        _meshID = _brg.RegisterMesh(_mesh);
        _materialID = _brg.RegisterMaterial(_material);

        // Create the buffer that holds our instance data
        var bufferCountForInstances = BufferCountForInstances(BytesPerInstance, (int) _instancesCount, ExtraBytes);
        _instanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
            bufferCountForInstances,
            sizeof(int));

        // Place one zero matrix at the start of the instance data buffer, so loads from address 0 will return zero
        var zero = new Matrix4x4[1] { Matrix4x4.zero };

        // Create transform matrices for our three example instances
        var matrices = new float4x4[_instancesCount];
        for (var i = 0; i < matrices.Length; i++)
        {
            matrices[i] = Matrix4x4.Translate(Random.onUnitSphere * _radius);
        }

        // Convert the transform matrices into the packed format expected by the shader
        _objectToWorld = new float3x4[_instancesCount];
        for (var i = 0; i < _objectToWorld.Length; i++)
        {
            _objectToWorld[i] = new float3x4(
                matrices[i].c0.x, matrices[i].c1.x, matrices[i].c2.x, matrices[i].c3.x,
                matrices[i].c0.y, matrices[i].c1.y, matrices[i].c2.y, matrices[i].c3.y,
                matrices[i].c0.z, matrices[i].c1.z, matrices[i].c2.z, matrices[i].c3.z
            );
        }

        // Also create packed inverse matrices
        _worldToObject = new float3x4[_instancesCount];
        for (var i = 0; i < _worldToObject.Length; i++)
        {
            var inverse = math.inverse(matrices[i]);
            _worldToObject[i] = new float3x4(
                inverse.c0.x, inverse.c1.x, inverse.c2.x, inverse.c3.x,
                inverse.c0.y, inverse.c1.y, inverse.c2.y, inverse.c3.y,
                inverse.c0.z, inverse.c1.z, inverse.c2.z, inverse.c3.z
            );
        }

        // In case of _instancesCount = 1, the instance data is placed into the buffer like this:
        // Offset | Description
        //      0 | 64 bytes of zeroes, so loads from address 0 return zeroes
        //     64 | 32 uninitialized bytes to make working with SetData easier, otherwise unnecessary
        //     96 | unity_ObjectToWorld, float3x4 matrix
        //    144 | unity_WorldToObject, float3x4 matrix

        // Compute start addresses for the different instanced properties. unity_ObjectToWorld starts
        // at address 96 instead of 64, because the computeBufferStartIndex parameter of SetData
        // is expressed as source array elements, so it is easier to work in multiples of sizeof(PackedMatrix).
        _byteAddressObjectToWorld = SizeOfPackedMatrix * 2;
        _byteAddressWorldToObject = _byteAddressObjectToWorld + SizeOfPackedMatrix * _instancesCount;
        // Upload our instance data to the GraphicsBuffer, from where the shader can load them.
        _instanceData.SetData(zero, 0, 0, 1);
        _instanceData.SetData(_objectToWorld, 0, (int) (_byteAddressObjectToWorld / SizeOfPackedMatrix),
            _objectToWorld.Length);
        _instanceData.SetData(_worldToObject, 0, (int) (_byteAddressWorldToObject / SizeOfPackedMatrix),
            _worldToObject.Length);

        // Set up metadata values to point to the instance data. Set the most significant bit 0x80000000 in each,
        // which instructs the shader that the data is an array with one value per instance, indexed by the instance index.
        // Any metadata values used by the shader and not set here will be zero. When such a value is used with
        // UNITY_ACCESS_DOTS_INSTANCED_PROP (i.e. without a default), the shader will interpret the
        // 0x00000000 metadata value so that the value will be loaded from the start of the buffer, which is
        // where we uploaded the matrix "zero" to, so such loads are guaranteed to return zero, which is a reasonable
        // default value.
        var metadata = new NativeArray<MetadataValue>(2, Allocator.Temp)
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
            }
        };

        // Finally, create a batch for our instances, and make the batch use the GraphicsBuffer with our
        // instance data, and the metadata values that specify where the properties are. Note that
        // we do not need to pass any batch size here.
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

    // We need to dispose our GraphicsBuffer and BatchRendererGroup when our script is no longer used,
    // to avoid leaking anything. Registered Meshes and Materials, and any batches added to the
    // BatchRendererGroup are automatically disposed when disposing the BatchRendererGroup.
    private void OnDisable()
    {
        _instanceData.Dispose();
        _brg.Dispose();
    }

    // The callback method called by Unity whenever it visibility culls to determine which
    // objects to draw. This method will output draw commands that describe to Unity what
    // should be drawn for this BatchRendererGroup.
    private unsafe JobHandle OnPerformCulling(
        BatchRendererGroup rendererGroup,
        BatchCullingContext cullingContext,
        BatchCullingOutput cullingOutput,
        IntPtr userContext)
    {
        // UnsafeUtility.Malloc() requires an alignment, so use the largest integer type's alignment
        // which is a reasonable default.
        var alignment = UnsafeUtility.AlignOf<long>();

        // Acquire a pointer to the BatchCullingOutputDrawCommands struct so we can easily
        // modify it directly.
        var drawCommands = (BatchCullingOutputDrawCommands*) cullingOutput.drawCommands.GetUnsafePtr();

        // Allocate memory for the output arrays. In a more complicated implementation the amount of memory
        // allocated could be dynamically calculated based on what we determined to be visible.
        // In this example, we will just assume that all of our instances are visible and allocate
        // memory for each of them. We need the following allocations:
        // - a single draw command (which draws kNumInstances instances)
        // - a single draw range (which covers our single draw command)
        // - kNumInstances visible instance indices.
        // The arrays must always be allocated using Allocator.TempJob.
        drawCommands->drawCommands = (BatchDrawCommand*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>(),
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

        // Our example does not use depth sorting, so we can leave the instanceSortingPositions as null.
        drawCommands->instanceSortingPositions = null;
        drawCommands->instanceSortingPositionFloatCount = 0;

        // Configure our single draw command to draw kNumInstances instances
        // starting from offset 0 in the array, using the batch, material and mesh
        // IDs that we registered in the Start() method. No special flags are set.
        drawCommands->drawCommands[0].visibleOffset = 0;
        drawCommands->drawCommands[0].visibleCount = _instancesCount;
        drawCommands->drawCommands[0].batchID = _batchID;
        drawCommands->drawCommands[0].materialID = _materialID;
        drawCommands->drawCommands[0].meshID = _meshID;
        drawCommands->drawCommands[0].submeshIndex = 0;
        drawCommands->drawCommands[0].splitVisibilityMask = 0xff;
        drawCommands->drawCommands[0].flags = 0;
        drawCommands->drawCommands[0].sortingPosition = 0;

        // Configure our single draw range to cover our single draw command which
        // is at offset 0.
        drawCommands->drawRanges[0].drawCommandsBegin = 0;
        drawCommands->drawRanges[0].drawCommandsCount = 1;
        // In this example we don't care about shadows or motion vectors, so we leave everything
        // to the default zero values, except the renderingLayerMask which we have to set to all ones
        // so the instances will be drawn regardless of mask settings when rendering.
        drawCommands->drawRanges[0].filterSettings = new BatchFilterSettings { renderingLayerMask = 0xffffffff, };

        // Finally, write the actual visible instance indices to their array. In a more complicated
        // implementation, this output would depend on what we determined to be visible, but in this example
        // we will just assume that everything is visible.
        for (var i = 0; i < _instancesCount; ++i)
        {
            drawCommands->visibleInstances[i] = i;
        }

        // This simple example does not use jobs, so we can just return an empty JobHandle.
        // Performance sensitive applications are encouraged to use Burst jobs to implement
        // culling and draw command output, in which case we would return a handle here that
        // completes when those jobs have finished.
        return new JobHandle();
    }
}