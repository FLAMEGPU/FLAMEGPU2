import pytest
from unittest import TestCase
from pyflamegpu import *

class InitGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(10);
        for i in range(10):
            graph.setVertexID(i, i);
            graph.setVertexPropertyFloat("vertex_float", i, i);
            graph.setVertexPropertyDouble("vertex_double2", i, 0, i + 11);
            graph.setVertexPropertyDouble("vertex_double2", i, 1, i + 21);  # Redundant for Py
            graph.setVertexPropertyArrayInt("vertex_int3", i, [i + 1, i + 2, i + 3]);

        graph.setEdgeCount(20);
        for i in range(10):
            graph.setEdgeSource(i, int(i / 2));
            graph.setEdgeDestination(i, (i + 2) % 10);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [ i + 41, i + 42, i + 43]);

        for i in range(10, 20):
            graph.setEdgeSourceDestination(i, int(i / 2), (i + 6) % 10);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [ i + 41, i + 42, i + 43]);


# Init's same as InitGraph, however fills the vertices/edges with zero
class InitGraph2(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(10);
        for i in range(10):
            graph.setVertexID(i, i);
            graph.setVertexPropertyFloat("vertex_float", i, 0);
            graph.setVertexPropertyDouble("vertex_double2", i, 0, 0);
            graph.setVertexPropertyDouble("vertex_double2", i, 1, 0);  # Redundant for Py
            graph.setVertexPropertyArrayInt("vertex_int3", i, [0, 0, 0]);

        graph.setEdgeCount(20);
        for i in range(10):
            graph.setEdgeSource(i, i % 10);
            graph.setEdgeDestination(i, 0);
            graph.setEdgePropertyInt("edge_int", i, 0);
            graph.setEdgePropertyDouble("edge_double2", i, 0, 0);
            graph.setEdgePropertyDouble("edge_double2", i, 1, 0);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [0, 0, 0]);

        for i in range(10, 20):
            graph.setEdgeSourceDestination(i, i % 10, (2 * i + 4) % 10);
            graph.setEdgePropertyInt("edge_int", i, 0);
            graph.setEdgePropertyDouble("edge_double2", i, 0, 0);
            graph.setEdgePropertyDouble("edge_double2", i, 1, 0);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [0, 0, 0]);

# Alternate version to InitGraph
class InitGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(30);
        for i in range(30):
            graph.setVertexID(i, i);
            graph.setVertexPropertyFloat("vertex_float", i, i);
            graph.setVertexPropertyDouble("vertex_double2", i, 0, i + 11);
            graph.setVertexPropertyDouble("vertex_double2", i, 1, i + 21);  # Redundant for Py
            graph.setVertexPropertyArrayInt("vertex_int3", i, [i + 1, i + 2, i + 3]);

        graph.setEdgeCount(60);
        for i in range(30):
            graph.setEdgeSource(i, int(i / 2));
            graph.setEdgeDestination(i, (i + 2) % 30);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [i + 41, i + 52, i + 43]);

        for i in range(30, 60):
            graph.setEdgeSourceDestination(i, int(i / 2), (i + 18) % 30);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [i + 41, i + 52, i + 43]);

# Set graph to same data as InitGraph, it assumes vertice/edge counts are correct
class SetGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        for i in range(10):
            graph.setVertexID(i, i);
            graph.setVertexPropertyFloat("vertex_float", i, i);
            graph.setVertexPropertyDouble("vertex_double2", i, 0, i + 11);
            graph.setVertexPropertyDouble("vertex_double2", i, 1, i + 21);  # Redundant for Py
            graph.setVertexPropertyArrayInt("vertex_int3", i, [i + 1, i + 2, i + 3]);

        for i in range(10):
            graph.setEdgeSource(i, int(i / 2));
            graph.setEdgeDestination(i, (i + 2) % 10);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [i + 41, i + 42, i + 43]);

        for i in range(10, 20):
            graph.setEdgeSourceDestination(i, int(i / 2), (i + 6) % 10);
            graph.setEdgePropertyInt("edge_int", i, i + 70);
            graph.setEdgePropertyDouble("edge_double2", i, 0, i + 61);
            graph.setEdgePropertyDouble("edge_double2", i, 1, i + 51);  # Redundant for Py
            graph.setEdgePropertyArrayFloat("edge_float3", i, [i + 41, i + 42, i + 43]);

class HostCheckGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        # Vertices
        assert graph.getVertexCount() == 10;
        for i in range(10):
            assert graph.getVertexID(i) == i;
            assert graph.getVertexPropertyFloat("vertex_float", i) == i;
            assert graph.getVertexPropertyDouble("vertex_double2", i, 0) == i + 11;
            # assert graph.getVertexPropertyDouble("vertex_double2", i, 1) == i + 21;  # Redundant for Py
            assert graph.getVertexPropertyArrayInt("vertex_int3", i) == (i + 1, i + 2, i + 3);

        # Edges
        assert graph.getEdgeCount() == 20;
        for i in range(10):
            assert graph.getEdgeSource(i) == int(i / 2);
            assert graph.getEdgeDestination(i) == (i + 2) % 10;
            assert graph.getEdgePropertyInt("edge_int", i) == i + 70;
            assert graph.getEdgePropertyDouble("edge_double2", i, 0) == i + 61;
            # assert graph.getEdgePropertyDouble("edge_double2", i, 1) == i + 51;  # Redundant for Py
            assert graph.getEdgePropertyArrayFloat("edge_float3", i) == (i + 41, i + 42, i + 43);

        for i in range(10, 20):
            assert graph.getEdgeSourceDestination(i) == (int(i / 2), (i + 6) % 10 )
            assert graph.getEdgePropertyInt("edge_int", i) == i + 70;
            assert graph.getEdgePropertyDouble("edge_double2", i, 0) == i + 61;
            # assert graph.getEdgePropertyDouble("edge_double2", i, 1) == si + 51;  # Redundant for Py
            assert graph.getEdgePropertyArrayFloat("edge_float3", i) == (i + 41, i + 42, i + 43);

# Equivalent version to HostCheckGraph but for InitGraph3
class HostCheckGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        # Vertices
        assert graph.getVertexCount() == 30;
        for i in range(30):
            assert graph.getVertexID(i) == i;
            assert graph.getVertexPropertyFloat("vertex_float", i) == i;
            assert graph.getVertexPropertyDouble("vertex_double2", i, 0) == i + 11;
            # assert graph.getVertexPropertyDouble("vertex_double2", i, 1) == i + 21;  # Redundant for Py
            assert graph.getVertexPropertyArrayInt("vertex_int3", i) == (i + 1, i + 2, i + 3);

        # Edges
        assert graph.getEdgeCount() == 60;
        for i in range(30):
            assert graph.getEdgeSource(i) == int(i / 2);
            assert graph.getEdgeDestination(i) == (i + 2) % 30;
            assert graph.getEdgePropertyInt("edge_int", i) == i + 70;
            assert graph.getEdgePropertyDouble("edge_double2", i, 0) == i + 61;
            # assert graph.getEdgePropertyDouble("edge_double2", i, 1) == i + 51;  # Redundant for Py
            assert graph.getEdgePropertyArrayFloat("edge_float3", i) == (i + 41, i + 52, i + 43);

        for i in range(30, 60):
            assert graph.getEdgeSourceDestination(i) == (int(i / 2), (i + 18) % 30);
            assert graph.getEdgePropertyInt("edge_int", i) == i + 70;
            assert graph.getEdgePropertyDouble("edge_double2", i, 0) == i + 61;
            # assert graph.getEdgePropertyDouble("edge_double2", i, 1) == i + 51;  # Redundant for Py
            assert graph.getEdgePropertyArrayFloat("edge_float3", i) == (i + 41, i + 52, i + 43);

class HostException(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
    
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphName exception
            FLAMEGPU.environment.getDirectedGraph("does not exist");
        assert e.value.type() == "InvalidGraphName"
        graph = FLAMEGPU.environment.getDirectedGraph("graph");

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException exception
            graph.getVertexID(0)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException exception
            graph.getEdgeSource(0)
        assert e.value.type() == "OutOfBoundsException"
        
        graph.setVertexCount(10);
        graph.setEdgeCount(10);


        # Name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyFloat("does not exist", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.setVertexPropertyArrayDouble("does not exist", 0, 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyArrayInt("does not exist", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyInt("does not exist", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.setEdgePropertyArrayDouble("does not exist", 0, 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyArrayFloat("does not exist", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getVertexPropertyFloat("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.getVertexPropertyArrayDouble("does not exist", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getVertexPropertyArrayInt("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getEdgePropertyInt("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.getEdgePropertyArrayDouble("does not exist", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getEdgePropertyArrayFloat("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"

        # Type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyUInt("vertex_float", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.setVertexPropertyArrayUInt("vertex_double2", 0, 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyArrayUInt("vertex_int3", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyUInt("edge_int", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.setEdgePropertyArrayUInt("edge_double2", 0, 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyArrayUInt("edge_float3", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getVertexPropertyUInt("vertex_float", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.getVertexPropertyArrayUInt("vertex_double2", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getVertexPropertyArrayUInt("vertex_int3", 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getEdgePropertyUInt("edge_int", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     graph.getEdgePropertyArrayUInt("edge_double2", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getEdgePropertyArrayUInt("edge_float3", 0);
        assert e.value.type() == "InvalidGraphProperty"
        
        # Length
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyInt("vertex_int3", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyArrayInt("vertex_int3", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setVertexPropertyArrayInt("vertex_int3", 0, [1, 2]);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyFloat("edge_float3", 0, 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyArrayFloat("edge_float3", 0, []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.setEdgePropertyArrayFloat("edge_float3", 0, [1, 2]);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getVertexPropertyInt("vertex_int3", 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            graph.getEdgePropertyFloat("edge_float3", 0);
        assert e.value.type() == "InvalidGraphProperty"


        # Array Index
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setVertexPropertyArrayDouble("vertex_double2", 0, 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setEdgePropertyArrayDouble("edge_double2", 0, 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          #  graph.getVertexPropertyArrayDouble("vertex_double2", 0, 3);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          #   graph.getEdgePropertyArrayDouble("edge_double2", 0, 3);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setVertexPropertyArrayDouble("vertex_double2", 0, 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setEdgePropertyArrayDouble("edge_double2", 0, 3, 0);
        # assert e.value.type() == "OutOfBoundsException"

        # Vertex/Edge Index
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setVertexID(11, 0);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setVertexPropertyFloat("vertex_float", 11, 0);
        assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setVertexPropertyArrayDouble("vertex_double2", 11, 0, 0);
        # assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setVertexPropertyArrayInt("vertex_int3", 11, [1, 2, 3]);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setEdgeSource(11, 0);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setEdgeDestination(11, 0);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setEdgeSourceDestination(11, 0, 0);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setEdgePropertyInt("edge_int", 11, 0);
        assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # graph.setEdgePropertyArrayDouble("edge_double2", 11, 0, 0);
        # assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.setEdgePropertyArrayFloat("edge_float3", 11, [1, 2, 3]);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getVertexID(11);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getVertexPropertyFloat("vertex_float", 11);
        assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
        #     graph.getVertexPropertyArrayDouble("vertex_double2", 11, 0);
        # assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getVertexPropertyArrayInt("vertex_int3", 11);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getEdgeSource(11);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getEdgeDestination(11);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getEdgeSourceDestination(11);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getEdgePropertyInt("edge_int", 11);
        assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
        #    graph.getEdgePropertyArrayDouble("edge_double2", 11, 0);
        # assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            graph.getEdgePropertyArrayFloat("edge_float3", 11);
        assert e.value.type() == "OutOfBoundsException"
        
class HostDeviceCheckGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent").getPopulationData();
        for i in range(20):
            if i < 10:
                assert agent[i].getVariableID("vertex_id") == i;
                assert agent[i].getVariableFloat("vertex_float") == i;
                assert agent[i].getVariableArrayDouble("vertex_double2") == (i + 11, i + 21);
                assert agent[i].getVariableID("edge_dest") == (i + 2) % 10;
            else:
                assert agent[i].getVariableID("edge_dest") == (i + 6) % 10;
            # Edges
            assert agent[i].getVariableID("edge_source") == int(i / 2);
            assert agent[i].getVariableInt("edge_int") == i + 70;
            assert agent[i].getVariableArrayDouble("edge_double2") == (i + 61, i + 51);

class HostDeviceCheckGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent").getPopulationData();
        for i in range(60):
            if i < 30:
                assert agent[i].getVariableID("vertex_id") == i;
                assert agent[i].getVariableFloat("vertex_float") == i;
                assert agent[i].getVariableArrayDouble("vertex_double2") == (i + 11, i + 21);
                assert agent[i].getVariableID("edge_dest") == (i + 2) % 30;
            else:
                assert agent[i].getVariableID("edge_dest") == (i + 18) % 30;
            # Edges
            assert agent[i].getVariableID("edge_source") == int(i / 2);
            assert agent[i].getVariableInt("edge_int") == i + 70;
            assert agent[i].getVariableArrayDouble("edge_double2") == (i + 61, i + 51);

class always_exit(pyflamegpu.HostCondition):
    def run(self, FLAMEGPU):
        return pyflamegpu.EXIT;

class HostTestEdgesOut(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(5);
        graph.setEdgeCount(15);
        k = int(0);
        for i in range(5):
            for j in range(i + 1):
                graph.setEdgePropertyID("src_copy", k, j);
                graph.setEdgeSourceDestination(k, j, i);
                k += 1;
                
class HostTestEdgesIn(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(5);
        graph.setEdgeCount(15);
        k = int(0);
        for i in range(5):
            for j in range(i + 1):
                graph.setEdgePropertyID("dest_copy", k, j);
                graph.setEdgeSourceDestination(k, i, j);
                k += 1;

class EnvironmentDirectedGraphTest(TestCase):
    def test_HostGetResetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph().__disown__());
        # Init graph with different known data
        model.newLayer().addHostFunction(InitGraph3().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph3().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_HostSetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with junk data
        model.newLayer().addHostFunction(InitGraph2().__disown__());
        # Set the graphs data to known data
        model.newLayer().addHostFunction(SetGraph().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();
      
    def test_HostException(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with junk data
        model.newLayer().addHostFunction(HostException().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();
        
    
    CopyGraphToAgent1_func = """
    FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent1, flamegpu::MessageNone, flamegpu::MessageNone) {
        if (FLAMEGPU->getID() <= 20u) {
            flamegpu::DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
            if (FLAMEGPU->getID() <= 10u) {
                FLAMEGPU->setVariable<flamegpu::id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
                // vertex_int3, device full array access not available, so skipped
            }
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_source", graph.getEdgeSource(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_dest", graph.getEdgeDestination(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
            // edge_float3, device full array access not available, so skipped
        }
        return flamegpu::ALIVE;
    }
    """
    
    CopyGraphToAgent3_func = """
    FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent3, flamegpu::MessageNone, flamegpu::MessageNone) {
        if (FLAMEGPU->getID() <= 60u) {
            flamegpu::DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
            if (FLAMEGPU->getID() <= 30u) {
                FLAMEGPU->setVariable<flamegpu::id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
                // vertex_int3, device full array access not available, so skipped
            }
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_source", graph.getEdgeSource(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_dest", graph.getEdgeDestination(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
            // edge_float3, device full array access not available, so skipped
        }
        return flamegpu::ALIVE;
    }
    """
    
    def test_DeviceGetResetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        agent = model.newAgent("agent");

        agent.newVariableID("vertex_id");
        agent.newVariableFloat("vertex_float");
        agent.newVariableArrayDouble("vertex_double2", 2);
        # agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        agent.newVariableID("edge_source");
        agent.newVariableID("edge_dest");
        agent.newVariableInt("edge_int");
        agent.newVariableArrayDouble("edge_double2", 2);
        # agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped
        CopyGraphToAgent1 = agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);
        CopyGraphToAgent3 = agent.newRTCFunction("fn2", self.CopyGraphToAgent3_func);

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());
        # Copy Data from Graph to Agent
        model.newLayer().addAgentFunction(CopyGraphToAgent1);
        # Check the agent data is correct persists
        model.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());
        # Init graph with different known data
        model.newLayer().addHostFunction(InitGraph3().__disown__());
        # Copy Data from Graph to Agent
        model.newLayer().addAgentFunction(CopyGraphToAgent3);
        # Check the agent data is correct persists
        model.newLayer().addHostFunction(HostDeviceCheckGraph3().__disown__());

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 60);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

    def test_SubModelSet_MasterModelHostGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");
        submodel.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        # Init graph with known data
        submodel.newLayer().addHostFunction(InitGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();

        # Init graph with known data
        model.newLayer().addSubModel(sub_desc);

        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_SubModelSet_MasterModelDeviceGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");

        agent = submodel.newAgent("agent");
        agent.newVariableFloat("foobar");  # Agent is not required in this model for this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        submodel.newLayer().addHostFunction(InitGraph().__disown__());

        # Copy data to agent and check in host fn
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        master_agent = model.newAgent("agent");
        master_agent.newVariableFloat("foobar");  # Agent is only used to init a population
        master_agent.newVariableID("vertex_id");
        master_agent.newVariableFloat("vertex_float");
        master_agent.newVariableArrayDouble("vertex_double2", 2);
        # master_agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        master_agent.newVariableID("edge_source");
        master_agent.newVariableID("edge_dest");
        master_agent.newVariableInt("edge_int");
        master_agent.newVariableArrayDouble("edge_double2", 2);
        # master_agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped

        CopyGraphToAgent1 = master_agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();
        sub_desc.bindAgent("agent", "agent");

        # Init graph with known data
        model.newLayer().addSubModel(sub_desc);

        # Check the data persists
        model.newLayer().addAgentFunction(CopyGraphToAgent1);
        model.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(master_agent, 20);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

    def test_MasterModelSet_SubModelHostGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");
        submodel.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        # Init graph with known data
        submodel.newLayer().addHostFunction(HostCheckGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());

        # Check the data persists
        model.newLayer().addSubModel(sub_desc);

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_MasterModelSet_SubModelDeviceGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");

        agent = submodel.newAgent("agent");
        agent.newVariableID("vertex_id");
        agent.newVariableFloat("vertex_float");
        agent.newVariableArrayDouble("vertex_double2", 2);
        # agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        agent.newVariableID("edge_source");
        agent.newVariableID("edge_dest");
        agent.newVariableInt("edge_int");
        agent.newVariableArrayDouble("edge_double2", 2);
        # agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped

        CopyGraphToAgent1 = agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);


        # Copy data to agent and check in host fn
        submodel.newLayer().addAgentFunction(CopyGraphToAgent1);
        submodel.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        master_agent = model.newAgent("agent");
        master_agent.newVariableFloat("foobar");  # Agent is only used to init a population

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();
        sub_desc.bindAgent("agent", "agent");

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());

        # Check the data persists
        model.newLayer().addSubModel(sub_desc);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(master_agent, 20);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();
        
    EdgesOut_func = """
    FLAMEGPU_AGENT_FUNCTION(IterateEdges, flamegpu::MessageNone, flamegpu::MessageNone) {
        flamegpu::id_t src = FLAMEGPU->getIndex();
        unsigned int ct = 0;
        bool src_all_correct = true;
        auto filter = FLAMEGPU->environment.getDirectedGraph("graph").outEdges(src);
        FLAMEGPU->setVariable<int>("count2", filter.size());
        for (auto &edge : filter) {
            src_all_correct &= edge.getProperty<flamegpu::id_t>("src_copy") == src;
            FLAMEGPU->setVariable<flamegpu::id_t, 5>("dests", ct, edge.getEdgeDestination());
            ++ct;
        }
        FLAMEGPU->setVariable<int>("count", ct);
        FLAMEGPU->setVariable<int>("src_all_correct", src_all_correct ? 1 : 0);
        return flamegpu::ALIVE;
    }
    """
        
    def test_EdgesOut(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newEdgePropertyID("src_copy");

        agent = model.newAgent("agent");
        agent.newVariableArrayID("dests", 5);
        agent.newVariableInt("count");
        agent.newVariableInt("count2");
        agent.newVariableInt("src_all_correct");
        EdgesOut = agent.newRTCFunction("iterate_edges", self.EdgesOut_func);

        # Init graph with known data
        model.newLayer().addHostFunction(HostTestEdgesOut());
        model.newLayer().addAgentFunction(EdgesOut);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 5);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        k = int(0);
        for agt in pop:
            assert agt.getVariableInt("src_all_correct") == 1;
            assert agt.getVariableInt("count") == 5 - k;
            assert agt.getVariableInt("count2") == 5 - k;
            for i in range(5 - k):
                assert agt.getVariableID("dests", i) == k + i;
            k += 1;
            
    EdgesIn_func = """
    FLAMEGPU_AGENT_FUNCTION(IterateEdges, flamegpu::MessageNone, flamegpu::MessageNone) {
        flamegpu::id_t dest = FLAMEGPU->getIndex();
        unsigned int ct = 0;
        bool dest_all_correct = true;
        auto filter = FLAMEGPU->environment.getDirectedGraph("graph").inEdges(dest);
        FLAMEGPU->setVariable<int>("count2", filter.size());
        for (auto& edge : filter) {
            dest_all_correct &= edge.getProperty<flamegpu::id_t>("dest_copy") == dest;
            FLAMEGPU->setVariable<flamegpu::id_t, 5>("srcs", ct, edge.getEdgeSource());
            ++ct;
        }
        FLAMEGPU->setVariable<int>("count", ct);
        FLAMEGPU->setVariable<int>("dest_all_correct", dest_all_correct ? 1 : 0);
        return flamegpu::ALIVE;
    }
    """
    
    def test_EdgesIn(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newEdgePropertyID("dest_copy");

        agent = model.newAgent("agent");
        agent.newVariableArrayID("srcs", 5);
        agent.newVariableInt("count");
        agent.newVariableInt("count2");
        agent.newVariableInt("dest_all_correct");
        EdgesIn = agent.newRTCFunction("iterate_edges", self.EdgesIn_func);

        # Init graph with known data
        model.newLayer().addHostFunction(HostTestEdgesIn());
        model.newLayer().addAgentFunction(EdgesIn);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 5);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        k = int(0);
        for agt in pop:
            assert agt.getVariableInt("dest_all_correct") == 1;
            assert agt.getVariableInt("count") == 5 - k;
            assert agt.getVariableInt("count2") == 5 - k;
            for i in range(5 - k):
                assert agt.getVariableID("srcs", i) == k + i;
            k += 1;
