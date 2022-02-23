using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

// This example demonstrates how to write a very minimal BatchRendererGroup
// based custom renderer using the Universal Render Pipeline to help
// getting started with using BatchRendererGroup.
public class BatchRenderer : MonoBehaviour
{
    // Set this to a suitable Mesh via the Inspector, such as a Cube mesh
    [SerializeField] private Mesh _mesh;

    // Set this to a suitable Material via the Inspector, such as a default material that
    // uses Universal Render Pipeline/Lit
    [SerializeField] private Material _material;
    [SerializeField] private float _radius = 10;

    private BatchRendererGroup _brg;
    private GraphicsBuffer _instanceData;
    private BatchID _batchID;
    private BatchMeshID _meshID;
    private BatchMaterialID _materialID;

    // Some helper constants to make calculations later a bit more convenient.
    private const int SizeOfMatrix = sizeof(float) * 4 * 4;
    private const int SizeOfPackedMatrix = sizeof(float) * 4 * 3;
    private const int SizeOfFloat4 = sizeof(float) * 4;
    private const int BytesPerInstance = (SizeOfPackedMatrix * 2) + SizeOfFloat4;
    private const int ExtraBytes = SizeOfMatrix * 2;
    private const int NumInstances = 1000000;
    private const uint ByteAddressObjectToWorld = SizeOfPackedMatrix * 2;
    private const uint ByteAddressWorldToObject = ByteAddressObjectToWorld + SizeOfPackedMatrix * NumInstances;
    private const uint ByteAddressColor = ByteAddressWorldToObject + SizeOfPackedMatrix * NumInstances;

    // Unity provided shaders such as Universal Render Pipeline/Lit expect
    // unity_ObjectToWorld and unity_WorldToObject in a special packed 48 byte
    // format when the DOTS_INSTANCING_ON keyword is enabled.
    // This saves both GPU memory and GPU bandwidth.
    // We define a convenience type here so we can easily convert into this format.
    private struct PackedMatrix
    {
        public float m00;
        public float m10;
        public float m20;
        public float m01;
        public float m11;
        public float m21;
        public float m02;
        public float m12;
        public float m22;
        public float m03;
        public float m13;
        public float m23;

        public PackedMatrix(Matrix4x4 m)
        {
            m00 = m.m00;
            m10 = m.m10;
            m20 = m.m20;
            m01 = m.m01;
            m11 = m.m11;
            m21 = m.m21;
            m02 = m.m02;
            m12 = m.m12;
            m22 = m.m22;
            m03 = m.m03;
            m13 = m.m13;
            m23 = m.m23;
        }
    }

    // Raw buffers are allocated in ints, define an utility method to compute the required
    // amount of ints for our data.
    private static int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
    {
        // Round byte counts to int multiples
        bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        int totalBytes = bytesPerInstance * numInstances + extraBytes;
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
        _instanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
            BufferCountForInstances(BytesPerInstance, NumInstances, ExtraBytes),
            sizeof(int));


        // Place one zero matrix at the start of the instance data buffer, so loads from address 0 will return zero
        var zero = new Matrix4x4[1] { Matrix4x4.zero };

        // Create transform matrices for our three example instances
        var matrices = new Matrix4x4[NumInstances];

        for (int i = 0; i < NumInstances; i++)
        {
            var spawnPosition = Random.onUnitSphere * (_radius * 0.5f);
            matrices[i] = Matrix4x4.Translate(spawnPosition) *
                          Matrix4x4.LookAt(spawnPosition, Vector3.zero, Vector3.up);
        }

        // Convert the transform matrices into the packed format expected by the shader
        var objectToWorld = new PackedMatrix[NumInstances];

        for (int i = 0; i < NumInstances; i++)
        {
            objectToWorld[i] = new PackedMatrix(matrices[i]);
        }

        // Also create packed inverse matrices
        var worldToObject = new PackedMatrix[NumInstances];

        for (int i = 0; i < NumInstances; i++)
        {
            worldToObject[i] = new PackedMatrix(matrices[i].inverse);
        }

        // Make all instances have unique colors
        var colors = new Vector4[NumInstances];
        for (int i = 0; i < NumInstances; i++)
        {
            colors[i] = new Vector4(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 1);
        }
        // In this simple example, the instance data is placed into the buffer like this:
        // Offset | Description
        //      0 | 64 bytes of zeroes, so loads from address 0 return zeroes
        //     64 | 32 uninitialized bytes to make working with SetData easier, otherwise unnecessary
        //     96 | unity_ObjectToWorld, three packed float3x4 matrices
        //    240 | unity_WorldToObject, three packed float3x4 matrices
        //    384 | _BaseColor, three float4s

        // Compute start addresses for the different instanced properties. unity_ObjectToWorld starts
        // at address 96 instead of 64, because the computeBufferStartIndex parameter of SetData
        // is expressed as source array elements, so it is easier to work in multiples of sizeof(PackedMatrix).


        // Upload our instance data to the GraphicsBuffer, from where the shader can load them.
        _instanceData.SetData(zero, 0, 0, 1);
        _instanceData.SetData(objectToWorld, 0, (int) (ByteAddressObjectToWorld / SizeOfPackedMatrix),
            objectToWorld.Length);
        _instanceData.SetData(worldToObject, 0, (int) (ByteAddressWorldToObject / SizeOfPackedMatrix),
            worldToObject.Length);
        _instanceData.SetData(colors, 0, (int) (ByteAddressColor / SizeOfFloat4), colors.Length);

        // Set up metadata values to point to the instance data. Set the most significant bit 0x80000000 in each,
        // which instructs the shader that the data is an array with one value per instance, indexed by the instance index.
        // Any metadata values used by the shader and not set here will be zero. When such a value is used with
        // UNITY_ACCESS_DOTS_INSTANCED_PROP (i.e. without a default), the shader will interpret the
        // 0x00000000 metadata value so that the value will be loaded from the start of the buffer, which is
        // where we uploaded the matrix "zero" to, so such loads are guaranteed to return zero, which is a reasonable
        // default value.
        var metadata = new NativeArray<MetadataValue>(3, Allocator.Temp);
        metadata[0] = new MetadataValue
            { NameID = Shader.PropertyToID("unity_ObjectToWorld"), Value = 0x80000000 | ByteAddressObjectToWorld, };
        metadata[1] = new MetadataValue
            { NameID = Shader.PropertyToID("unity_WorldToObject"), Value = 0x80000000 | ByteAddressWorldToObject, };
        metadata[2] = new MetadataValue
            { NameID = Shader.PropertyToID("_BaseColor"), Value = 0x80000000 | ByteAddressColor, };

        // Finally, create a batch for our instances, and make the batch use the GraphicsBuffer with our
        // instance data, and the metadata values that specify where the properties are. Note that
        // we do not need to pass any batch size here.
        _batchID = _brg.AddBatch(metadata, _instanceData.bufferHandle);
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
        int alignment = UnsafeUtility.AlignOf<long>();

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
            (int*) UnsafeUtility.Malloc(NumInstances * sizeof(int), alignment, Allocator.TempJob);
        drawCommands->drawCommandPickingInstanceIDs = null;

        drawCommands->drawCommandCount = 1;
        drawCommands->drawRangeCount = 1;
        drawCommands->visibleInstanceCount = NumInstances;

        // Our example does not use depth sorting, so we can leave the instanceSortingPositions as null.
        drawCommands->instanceSortingPositions = null;
        drawCommands->instanceSortingPositionFloatCount = 0;

        // Configure our single draw command to draw kNumInstances instances
        // starting from offset 0 in the array, using the batch, material and mesh
        // IDs that we registered in the Start() method. No special flags are set.
        drawCommands->drawCommands[0].visibleOffset = 0;
        drawCommands->drawCommands[0].visibleCount = NumInstances;
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
        for (int i = 0; i < NumInstances; ++i)
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