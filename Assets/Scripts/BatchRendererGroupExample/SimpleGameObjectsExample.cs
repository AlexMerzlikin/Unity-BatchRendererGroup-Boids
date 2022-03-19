using UnityEngine;
using Random = UnityEngine.Random;

public class SimpleGameObjectsExample : MonoBehaviour
{
    [SerializeField] private Transform _prefab;

    [SerializeField] private float _motionSpeed;
    [SerializeField] private float _motionAmplitude;
    [SerializeField] private Vector3 _motionDirection;
    [SerializeField] private uint _instancesCount = 1;
    [SerializeField] private float _radius;


    private Transform[] _gameObjects;
    private float _phase;

    private void Start()
    {
        _gameObjects = new Transform[_instancesCount];
        for (var i = 0; i < _instancesCount; i++)
        {
           var position = Random.onUnitSphere * _radius;
           _gameObjects[i] = Instantiate(_prefab, position, Quaternion.identity);
        }
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
            _gameObjects[i].position += pos;
        }
    }
}