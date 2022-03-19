using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

// This example demonstrates how to write a very minimal BatchRendererGroup
// based custom renderer using the Universal Render Pipeline to help
// getting started with using BatchRendererGroup.
public class SimpleBRGVector4Example : MonoBehaviour
{
    // Set this to a suitable Mesh via the Inspector, such as a Cube mesh
    public Mesh _mesh;

    // Set this to a suitable Material via the Inspector, such as a default material that
    // uses Universal Render Pipeline/Lit
    private Material _material;

    private BatchRendererGroup m_BRG;

    private GraphicsBuffer m_InstanceData;
    private BatchID m_BatchID;
    private BatchMeshID m_MeshID;
    private BatchMaterialID m_MaterialID;

    // Some helper constants to make calculations later a bit more convenient.
    private const int kSizeOfMatrix = sizeof(float) * 4 * 4;
    private const int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
    private const int kSizeOfFloat4 = sizeof(float) * 4;
    private const int kBytesPerInstance = (kSizeOfPackedMatrix * 2);
    private const int kExtraBytes = kSizeOfMatrix;
    private const int kNumInstances = 1;

    // Raw buffers are allocated in ints, define an utility method to compute the required
    // amount of ints for our data.
    int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
    {
        // Round byte counts to int multiples
        bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
        int totalBytes = bytesPerInstance * numInstances + extraBytes;
        return totalBytes / sizeof(int);
    }

    // During initialization, we will allocate all required objects, and set up our custom instance data.
    void Start()
    {
        // Create the BatchRendererGroup and register assets
        m_BRG = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
        m_MeshID = m_BRG.RegisterMesh(_mesh);
        m_MaterialID = m_BRG.RegisterMaterial(_material);

        // Create the buffer that holds our instance data
        var bufferCountForInstances = BufferCountForInstances(kBytesPerInstance, kNumInstances, kExtraBytes);
        m_InstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
            bufferCountForInstances,
            sizeof(int));

        // Place one zero matrix at the start of the instance data buffer, so loads from address 0 will return zero
        var zero = new Matrix4x4[1] { Matrix4x4.zero };

        // Create transform matrices for our three example instances
        var matrices = new float4x4[kNumInstances] { Translate(new Vector3(2, 0, 0)), };

        // Convert the transform matrices into the packed format expected by the shader
        var objectToWorld = new float3x4[kNumInstances]
        {
            new(
                matrices[0].c0.x,
                matrices[0].c1.x,
                matrices[0].c2.x,
                matrices[0].c0.y,
                matrices[0].c1.y,
                matrices[0].c2.y,
                matrices[0].c0.z,
                matrices[0].c1.z,
                matrices[0].c2.z,
                matrices[0].c0.w,
                matrices[0].c1.w,
                matrices[0].c2.w)
        };

        // Also create packed inverse matrices
        var inverse = math.inverse(matrices[0]);
        var worldToObject = new float3x4[kNumInstances]
        {
            new(
                inverse.c0.x,
                inverse.c1.x,
                inverse.c2.x,
                inverse.c0.y,
                inverse.c1.y,
                inverse.c2.y,
                inverse.c0.z,
                inverse.c1.z,
                inverse.c2.z,
                inverse.c0.w,
                inverse.c1.w,
                inverse.c2.w),
        };


        // In this simple example, the instance data is placed into the buffer like this:
        // Offset | Description
        //      0 | 64 bytes of zeroes, so loads from address 0 return zeroes
        //     64 | unity_ObjectToWorld, packed float3x4 matrices
        //    112 | unity_WorldToObject, packed float3x4 matrices

        // Compute start addresses for the different instanced properties. unity_ObjectToWorld starts
        // at address 96 instead of 64, because the computeBufferStartIndex parameter of SetData
        // is expressed as source array elements, so it is easier to work in multiples of sizeof(PackedMatrix).
        uint byteAddressObjectToWorld = kSizeOfPackedMatrix;
        uint byteAddressWorldToObject = byteAddressObjectToWorld + kSizeOfPackedMatrix * kNumInstances;

        // Upload our instance data to the GraphicsBuffer, from where the shader can load them.
        m_InstanceData.SetData(zero, 0, 0, 1);
        m_InstanceData.SetData(objectToWorld, 0, (int)(byteAddressObjectToWorld / kSizeOfPackedMatrix),
            objectToWorld.Length);
        m_InstanceData.SetData(worldToObject, 0, (int)(byteAddressWorldToObject / kSizeOfPackedMatrix),
            worldToObject.Length);

        // Set up metadata values to point to the instance data. Set the most significant bit 0x80000000 in each,
        // which instructs the shader that the data is an array with one value per instance, indexed by the instance index.
        // Any metadata values used by the shader and not set here will be zero. When such a value is used with
        // UNITY_ACCESS_DOTS_INSTANCED_PROP (i.e. without a default), the shader will interpret the
        // 0x00000000 metadata value so that the value will be loaded from the start of the buffer, which is
        // where we uploaded the matrix "zero" to, so such loads are guaranteed to return zero, which is a reasonable
        // default value.
        var metadata = new NativeArray<MetadataValue>(2, Allocator.Temp);
        metadata[0] = new MetadataValue
        {
            NameID = Shader.PropertyToID("unity_ObjectToWorld"), Value = 0x80000000 | byteAddressObjectToWorld,
        };
        metadata[1] = new MetadataValue
        {
            NameID = Shader.PropertyToID("unity_WorldToObject"), Value = 0x80000000 | byteAddressWorldToObject,
        };

        // Finally, create a batch for our instances, and make the batch use the GraphicsBuffer with our
        // instance data, and the metadata values that specify where the properties are. Note that
        // we do not need to pass any batch size here.
        m_BatchID = m_BRG.AddBatch(metadata, m_InstanceData.bufferHandle);
    }

    // We need to dispose our GraphicsBuffer and BatchRendererGroup when our script is no longer used,
    // to avoid leaking anything. Registered Meshes and Materials, and any batches added to the
    // BatchRendererGroup are automatically disposed when disposing the BatchRendererGroup.
    private void OnDisable()
    {
        m_InstanceData.Dispose();
        m_BRG.Dispose();
    }

    // The callback method called by Unity whenever it visibility culls to determine which
    // objects to draw. This method will output draw commands that describe to Unity what
    // should be drawn for this BatchRendererGroup.
    public unsafe JobHandle OnPerformCulling(
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
        var drawCommands = (BatchCullingOutputDrawCommands*)cullingOutput.drawCommands.GetUnsafePtr();

        // Allocate memory for the output arrays. In a more complicated implementation the amount of memory
        // allocated could be dynamically calculated based on what we determined to be visible.
        // In this example, we will just assume that all of our instances are visible and allocate
        // memory for each of them. We need the following allocations:
        // - a single draw command (which draws kNumInstances instances)
        // - a single draw range (which covers our single draw command)
        // - kNumInstances visible instance indices.
        // The arrays must always be allocated using Allocator.TempJob.
        drawCommands->drawCommands = (BatchDrawCommand*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>(),
            alignment, Allocator.TempJob);
        drawCommands->drawRanges =
            (BatchDrawRange*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(), alignment, Allocator.TempJob);
        drawCommands->visibleInstances =
            (int*)UnsafeUtility.Malloc(kNumInstances * sizeof(int), alignment, Allocator.TempJob);
        drawCommands->drawCommandPickingInstanceIDs = null;

        drawCommands->drawCommandCount = 1;
        drawCommands->drawRangeCount = 1;
        drawCommands->visibleInstanceCount = kNumInstances;

        // Our example does not use depth sorting, so we can leave the instanceSortingPositions as null.
        drawCommands->instanceSortingPositions = null;
        drawCommands->instanceSortingPositionFloatCount = 0;

        // Configure our single draw command to draw kNumInstances instances
        // starting from offset 0 in the array, using the batch, material and mesh
        // IDs that we registered in the Start() method. No special flags are set.
        drawCommands->drawCommands[0].visibleOffset = 0;
        drawCommands->drawCommands[0].visibleCount = kNumInstances;
        drawCommands->drawCommands[0].batchID = m_BatchID;
        drawCommands->drawCommands[0].materialID = m_MaterialID;
        drawCommands->drawCommands[0].meshID = m_MeshID;
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
        for (int i = 0; i < kNumInstances; ++i)
            drawCommands->visibleInstances[i] = i;

        // This simple example does not use jobs, so we can just return an empty JobHandle.
        // Performance sensitive applications are encouraged to use Burst jobs to implement
        // culling and draw command output, in which case we would return a handle here that
        // completes when those jobs have finished.
        return new JobHandle();
    }

    private static float4x4 Translate(Vector3 vector) =>
        new(
            1f, 0.0f, 0.0f, vector.x,
            0.0f, 1f, 0.0f, vector.y,
            0.0f, 0.0f, 1f, vector.z,
            0.0f, 0.0f, 0.0f, 1f);
}