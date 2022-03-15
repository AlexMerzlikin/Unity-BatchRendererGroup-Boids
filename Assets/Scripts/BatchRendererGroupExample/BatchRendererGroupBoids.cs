using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

public class BatchRendererGroupBoids : MonoBehaviour
{
    [SerializeField] private int itemGridSize = 30;
    [SerializeField] private Mesh m_mesh;
    [SerializeField] private Material m_material;
    [SerializeField] private float m_motionSpeed = 3.0f;
    [SerializeField] private float m_motionAmplitude = 2.0f;
    [SerializeField] private float m_spacingFactor = 1.0f;

    private BatchRendererGroup m_BatchRendererGroup;
    private GraphicsBuffer m_GPUPersistentInstanceData;
    private NativeArray<Vector4> m_sysmemBuffer;
    private BatchID m_batchID;
    private BatchMaterialID m_materialID;
    private BatchMeshID m_meshID;
    private int m_itemCount;
    private bool m_initialized;
    private float m_phase;

    private void Start()
    {
        m_BatchRendererGroup = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
        m_itemCount = itemGridSize * itemGridSize;

        // Bounds
        var bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
        m_BatchRendererGroup.SetGlobalBounds(bounds);

        // Register mesh and material
        if (m_mesh)
        {
            m_meshID = m_BatchRendererGroup.RegisterMesh(m_mesh);
        }

        if (m_material)
        {
            m_materialID = m_BatchRendererGroup.RegisterMaterial(m_material);
        }

        // Batch metadata buffer
        int objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
        int matrixPreviousMID = Shader.PropertyToID("unity_MatrixPreviousM");
        int worldToObjectID = Shader.PropertyToID("unity_WorldToObject");
        int colorID = Shader.PropertyToID("_BaseColor");

        // Generate a grid of objects...
        int bigDataBufferVector4Count =
            4 + m_itemCount * (3 * 3 + 1); // 4xfloat4 zero + per instance = { 3x mat4x3, 1x float4 color }
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
        UpdatePositions(Vector3.zero);

        // Colors
        int colorOffset = positionOffset + m_itemCount * 3 * 3;
        for (int i = 0; i < m_itemCount; i++)
        {
            Color col = Color.HSVToRGB((i / (float) m_itemCount) % 1.0f, 1.0f, 1.0f);

            // write colors right after the 4x3 matrices
            m_sysmemBuffer[colorOffset + i] = new Vector4(col.r, col.g, col.b, 1.0f);
        }

        m_GPUPersistentInstanceData.SetData(m_sysmemBuffer);

        var batchMetadata = new NativeArray<MetadataValue>(4, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
        {
            [0] = CreateMetadataValue(objectToWorldID, 64, true),
            [1] = CreateMetadataValue(matrixPreviousMID, 64 + m_itemCount * UnsafeUtility.SizeOf<Vector4>() * 3,
                true),
            [2] = CreateMetadataValue(worldToObjectID, 64 + m_itemCount * UnsafeUtility.SizeOf<Vector4>() * 3 * 2,
                true),
            [3] = CreateMetadataValue(colorID, 64 + m_itemCount * UnsafeUtility.SizeOf<Vector4>() * 3 * 3, true)
        };
        // matrices
        // previous matrices
        // inverse matrices
        // colors

        // Register batch
        m_batchID = m_BatchRendererGroup.AddBatch(batchMetadata, m_GPUPersistentInstanceData.bufferHandle);

        m_initialized = true;
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

        drawCommands.visibleInstances = Malloc<int>(m_itemCount);
        int n = 0;
        for (int r = 0; r < itemGridSize; r++)
        {
            for (int i = 0; i < itemGridSize; i++)
            {
                drawCommands.visibleInstances[n++] = r * itemGridSize + i;
            }
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

    private void UpdatePositions(Vector3 pos)
    {
        int positionOffset = 4;
        int itemCountOffset = itemGridSize * itemGridSize * 3; // 3xfloat4 per matrix
        for (int z = 0; z < itemGridSize; z++)
        {
            for (int x = 0; x < itemGridSize; x++)
            {
                float px = (x - itemGridSize / 2) * m_spacingFactor;
                float pz = (z - itemGridSize / 2) * m_spacingFactor;
                int i = z * itemGridSize + x;

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
                m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset] =
                    m_sysmemBuffer[positionOffset + i * 3 + 0];
                m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset] =
                    m_sysmemBuffer[positionOffset + i * 3 + 1];
                m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset] =
                    m_sysmemBuffer[positionOffset + i * 3 + 2];

                // compute the new current frame matrix
                m_sysmemBuffer[positionOffset + i * 3 + 0] = new Vector4(1, 0, 0, 0);
                m_sysmemBuffer[positionOffset + i * 3 + 1] = new Vector4(1, 0, 0, 0);
                m_sysmemBuffer[positionOffset + i * 3 + 2] = new Vector4(1, px + pos.x, pos.y, pz + pos.z);

                // compute the new inverse matrix
                m_sysmemBuffer[positionOffset + i * 3 + 0 + itemCountOffset * 2] = new Vector4(1, 0, 0, 0);
                m_sysmemBuffer[positionOffset + i * 3 + 1 + itemCountOffset * 2] = new Vector4(1, 0, 0, 0);
                m_sysmemBuffer[positionOffset + i * 3 + 2 + itemCountOffset * 2] =
                    new Vector4(1, -(px + pos.x), -pos.y, -(pz + pos.z));
            }
        }
    }

    // Update is called once per frame
    private void Update()
    {
        m_phase += Time.fixedDeltaTime * m_motionSpeed;

        if (m_motionAmplitude > 0.0f)
        {
            Vector3 pos = new Vector3(0, 0, Mathf.Cos(m_phase) * m_motionAmplitude);
            UpdatePositions(pos);
            // upload the full buffer
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer);
        }
    }

    private void OnDestroy()
    {
        if (!m_initialized)
        {
            return;
        }

        m_BatchRendererGroup.RemoveBatch(m_batchID);
        if (m_material)
        {
            m_BatchRendererGroup.UnregisterMaterial(m_materialID);
        }

        if (m_mesh)
        {
            m_BatchRendererGroup.UnregisterMesh(m_meshID);
        }

        m_BatchRendererGroup.Dispose();
        m_GPUPersistentInstanceData.Dispose();
        m_sysmemBuffer.Dispose();
    }
}