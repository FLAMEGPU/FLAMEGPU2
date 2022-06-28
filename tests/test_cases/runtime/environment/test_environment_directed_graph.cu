/**
* Tests of EnvironmentDirectedGraph
*
* This could perhaps be split into multiple files, one per useful class (Description, Host, Device)
*/
#include <filesystem>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace test_environment_directed_graph {
FLAMEGPU_HOST_FUNCTION(InitGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(10);
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setVertexID(i, i);
        graph.setVertexProperty<float>("vertex_float", i, static_cast<float>(i));
        graph.setVertexProperty<double, 2>("vertex_double2", i, 0, static_cast<double>(i + 11));
        graph.setVertexProperty<double>("vertex_double2", i, 1, static_cast<double>(i + 21));
        graph.setVertexProperty<int, 3>("vertex_int3", i, {static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3)});
    }
    graph.setEdgeCount(20);
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=10, ecount=20)
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setEdgeSource(i, i / 2);
        graph.setEdgeDestination(i, (i + 2) % 10);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43)});
    }
    for (unsigned int i = 10; i < 20; ++i) {
        graph.setEdgeSourceDestination(i, i / 2, (i + 6) % 10);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43)});
    }
}
// Init's same as InitGraph, however fills the vertices/edges with zero
FLAMEGPU_HOST_FUNCTION(InitGraph2) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(10);
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setVertexID(i, i);
        graph.setVertexProperty<float>("vertex_float", i, static_cast<float>(0));
        graph.setVertexProperty<double, 2>("vertex_double2", i, 0, static_cast<double>(0));
        graph.setVertexProperty<double>("vertex_double2", i, 1, static_cast<double>(0));
        graph.setVertexProperty<int, 3>("vertex_int3", i, { static_cast<int>(0), static_cast<int>(0) , static_cast<int>(0) });
    }
    graph.setEdgeCount(20);
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setEdgeSource(i, i % 10);
        graph.setEdgeDestination(i, 0);
        graph.setEdgeProperty<int>("edge_int", i, 0);
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(0));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(0));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(0), static_cast<float>(0) , static_cast<float>(0) });
    }
    for (unsigned int i = 10; i < 20; ++i) {
        graph.setEdgeSourceDestination(i, i % 10, (2 * i + 4) % 10);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(0));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(0));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(0));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(0), static_cast<float>(0) , static_cast<float>(0) });
    }
}
// Alternate version to InitGraph
FLAMEGPU_HOST_FUNCTION(InitGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(30);
    for (unsigned int i = 0; i < 30; ++i) {
        graph.setVertexID(i, i);
        graph.setVertexProperty<float>("vertex_float", i, static_cast<float>(i));
        graph.setVertexProperty<double, 2>("vertex_double2", i, 0, static_cast<double>(i + 11));
        graph.setVertexProperty<double>("vertex_double2", i, 1, static_cast<double>(i + 21));
        graph.setVertexProperty<int, 3>("vertex_int3", i, { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) });
    }
    graph.setEdgeCount(60);
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=30, ecount=60)
    for (unsigned int i = 0; i < 30; ++i) {
        graph.setEdgeSource(i, i / 2);
        graph.setEdgeDestination(i, (i + 2) % 30);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 52) , static_cast<float>(i + 43) });
    }
    for (unsigned int i = 30; i < 60; ++i) {
        graph.setEdgeSourceDestination(i, i / 2, (i + 18) % 30);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 52) , static_cast<float>(i + 43) });
    }
}
// Set graph to same data as InitGraph, it assumes vertice/edge counts are correct
FLAMEGPU_HOST_FUNCTION(SetGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setVertexID(i, i);
        graph.setVertexProperty<float>("vertex_float", i, static_cast<float>(i));
        graph.setVertexProperty<double, 2>("vertex_double2", i, 0, static_cast<double>(i + 11));
        graph.setVertexProperty<double>("vertex_double2", i, 1, static_cast<double>(i + 21));
        graph.setVertexProperty<int, 3>("vertex_int3", i, { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) });
    }
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=10, ecount=20)
    for (unsigned int i = 0; i < 10; ++i) {
        graph.setEdgeSource(i, i / 2);
        graph.setEdgeDestination(i, (i + 2) % 10);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) });
    }
    for (unsigned int i = 10; i < 20; ++i) {
        graph.setEdgeSourceDestination(i, i / 2, (i + 6) % 10);
        graph.setEdgeProperty<int>("edge_int", i, static_cast<int>(i + 70));
        graph.setEdgeProperty<double, 2>("edge_double2", i, 0, static_cast<double>(i + 61));
        graph.setEdgeProperty<double>("edge_double2", i, 1, static_cast<double>(i + 51));
        graph.setEdgeProperty<float, 3>("edge_float3", i, { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) });
    }
}
FLAMEGPU_HOST_FUNCTION(HostCheckGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Vertices
    EXPECT_EQ(graph.getVertexCount(), 10u);
    for (unsigned int i = 0; i < 10; ++i) {
        EXPECT_EQ(graph.getVertexID(i), i);
        EXPECT_EQ(graph.getVertexProperty<float>("vertex_float", i), static_cast<float>(i));
        EXPECT_EQ((graph.getVertexProperty<double, 2>("vertex_double2", i, 0)), static_cast<double>(i + 11));
        EXPECT_EQ(graph.getVertexProperty<double>("vertex_double2", i, 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) };
        EXPECT_EQ((graph.getVertexProperty<int, 3>("vertex_int3", i)), result);
    }
    // Edges
    EXPECT_EQ(graph.getEdgeCount(), 20u);
    for (unsigned int i = 0; i < 10; ++i) {
        EXPECT_EQ(graph.getEdgeSource(i), i / 2);
        EXPECT_EQ(graph.getEdgeDestination(i), (i + 2) % 10);
        EXPECT_EQ(graph.getEdgeProperty<int>("edge_int", i), static_cast<int>(i + 70));
        EXPECT_EQ((graph.getEdgeProperty<double, 2>("edge_double2", i, 0)), static_cast<double>(i + 61));
        EXPECT_EQ(graph.getEdgeProperty<double>("edge_double2", i, 1), static_cast<double>(i + 51));
        std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((graph.getEdgeProperty<float, 3>("edge_float3", i)), result);
    }
    for (unsigned int i = 10; i < 20; ++i) {
        const std::pair<id_t, id_t> srcdst = { i / 2, (i + 6) % 10 };
        EXPECT_EQ(graph.getEdgeSourceDestination(i), srcdst);
        EXPECT_EQ(graph.getEdgeProperty<int>("edge_int", i), static_cast<int>(i + 70));
        EXPECT_EQ((graph.getEdgeProperty<double, 2>("edge_double2", i, 0)), static_cast<double>(i + 61));
        EXPECT_EQ(graph.getEdgeProperty<double>("edge_double2", i, 1), static_cast<double>(i + 51));
        const std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((graph.getEdgeProperty<float, 3>("edge_float3", i)), result);
    }
}
// Equivalent version to HostCheckGraph but for InitGraph3
FLAMEGPU_HOST_FUNCTION(HostCheckGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Vertices
    EXPECT_EQ(graph.getVertexCount(), 30u);
    for (unsigned int i = 0; i < 30; ++i) {
        EXPECT_EQ(graph.getVertexID(i), i);
        EXPECT_EQ(graph.getVertexProperty<float>("vertex_float", i), static_cast<float>(i));
        EXPECT_EQ((graph.getVertexProperty<double, 2>("vertex_double2", i, 0)), static_cast<double>(i + 11));
        EXPECT_EQ(graph.getVertexProperty<double>("vertex_double2", i, 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2), static_cast<int>(i + 3) };
        EXPECT_EQ((graph.getVertexProperty<int, 3>("vertex_int3", i)), result);
    }
    // Edges
    EXPECT_EQ(graph.getEdgeCount(), 60u);
    for (unsigned int i = 0; i < 30; ++i) {
        EXPECT_EQ(graph.getEdgeSource(i), i / 2);
        EXPECT_EQ(graph.getEdgeDestination(i), (i + 2) % 30);
        EXPECT_EQ(graph.getEdgeProperty<int>("edge_int", i), static_cast<int>(i + 70));
        EXPECT_EQ((graph.getEdgeProperty<double, 2>("edge_double2", i, 0)), static_cast<double>(i + 61));
        EXPECT_EQ(graph.getEdgeProperty<double>("edge_double2", i, 1), static_cast<double>(i + 51));
        std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 52), static_cast<float>(i + 43) };
        EXPECT_EQ((graph.getEdgeProperty<float, 3>("edge_float3", i)), result);
    }
    for (unsigned int i = 30; i < 60; ++i) {
        const std::pair<id_t, id_t> srcdst = { i / 2, (i + 18) % 30 };
        EXPECT_EQ(graph.getEdgeSourceDestination(i), srcdst);
        EXPECT_EQ(graph.getEdgeProperty<int>("edge_int", i), static_cast<int>(i + 70));
        EXPECT_EQ((graph.getEdgeProperty<double, 2>("edge_double2", i, 0)), static_cast<double>(i + 61));
        EXPECT_EQ(graph.getEdgeProperty<double>("edge_double2", i, 1), static_cast<double>(i + 51));
        const std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 52), static_cast<float>(i + 43) };
        EXPECT_EQ((graph.getEdgeProperty<float, 3>("edge_float3", i)), result);
    }
}
TEST(TestEnvironmentDirectedGraph, TestHostGetResetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph);
    // Init graph with different known data
    model.newLayer().addHostFunction(InitGraph3);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph3);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, TestHostSetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with junk data
    model.newLayer().addHostFunction(InitGraph2);
    // Set the graphs data to known data
    model.newLayer().addHostFunction(SetGraph);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_HOST_FUNCTION(HostException) {
    EXPECT_THROW(FLAMEGPU->environment.getDirectedGraph("does not exist"), exception::InvalidGraphName);
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");

    EXPECT_THROW(graph.getVertexID(0), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeSource(0), exception::OutOfBoundsException);

    EXPECT_NO_THROW(graph.setVertexCount(10));
    EXPECT_NO_THROW(graph.setEdgeCount(10));

    // Name
    EXPECT_THROW(graph.setVertexProperty<float>("does not exist", 0, static_cast<float>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<double, 2>("does not exist", 0, 0, static_cast<double>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setVertexProperty<double>("does not exist", 0, 0, static_cast<double>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<int, 3>("does not exist", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setEdgeProperty<int>("does not exist", 0, static_cast<int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<double, 2>("does not exist", 0, 0, static_cast<double>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setEdgeProperty<double>("does not exist", 0, 0, static_cast<double>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<float, 3>("does not exist", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getVertexProperty<float>("does not exist", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<double, 2>("does not exist", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getVertexProperty<double>("does not exist", 0, 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<int, 3>("does not exist", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getEdgeProperty<int>("does not exist", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<double, 2>("does not exist", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getEdgeProperty<double>("does not exist", 0, 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<float, 3>("does not exist", 0)), exception::InvalidGraphProperty);

    // Type
    EXPECT_THROW(graph.setVertexProperty<unsigned int>("vertex_float", 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<unsigned int, 2>("vertex_double2", 0, 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setVertexProperty<unsigned int>("vertex_double2", 0, 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<unsigned int, 3>("vertex_int3", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setEdgeProperty<unsigned int>("edge_int", 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<unsigned int, 2>("edge_double2", 0, 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setEdgeProperty<unsigned int>("edge_double2", 0, 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<unsigned int, 3>("edge_float3", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getVertexProperty<unsigned int>("vertex_float", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<unsigned int, 2>("vertex_double2", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getVertexProperty<unsigned int>("vertex_double2", 0, 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<unsigned int, 3>("vertex_int3", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getEdgeProperty<unsigned int>("edge_int", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<unsigned int, 2>("edge_double2", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getEdgeProperty<unsigned int>("edge_double2", 0, 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<unsigned int, 3>("edge_float3", 0)), exception::InvalidGraphProperty);

    // Length
    EXPECT_THROW(graph.setVertexProperty<int>("vertex_int3", 0, static_cast<int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<int, 2>("vertex_int3", 0, 0, static_cast<int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setVertexProperty<int, 2>("vertex_int3", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.setEdgeProperty<float>("edge_float3", 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<float, 2>("edge_float3", 0, 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.setEdgeProperty<float, 2>("edge_float3", 0, { })), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getVertexProperty<int>("vertex_int3", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<int, 2>("vertex_int3", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getVertexProperty<int, 2>("vertex_int3", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(graph.getEdgeProperty<float>("edge_float3", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<float, 2>("edge_float3", 0, 0)), exception::InvalidGraphProperty);
    EXPECT_THROW((graph.getEdgeProperty<float, 2>("edge_float3", 0)), exception::InvalidGraphProperty);

    // Array Index
    EXPECT_THROW(graph.setVertexProperty<double>("vertex_double2", 0, 3, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeProperty<double>("edge_double2", 0, 3, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getVertexProperty<double>("vertex_double2", 0, 3), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeProperty<double>("edge_double2", 0, 3), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setVertexProperty<double, 2>("vertex_double2", 0, 3, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setEdgeProperty<double, 2>("edge_double2", 0, 3, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getVertexProperty<double, 2>("vertex_double2", 0, 3)), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getEdgeProperty<double, 2>("edge_double2", 0, 3)), exception::OutOfBoundsException);

    // Vertex/Edge Index
    EXPECT_THROW(graph.setVertexID(11, 0), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setVertexProperty<float>("vertex_float", 11, static_cast<float>(0)), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setVertexProperty<double, 2>("vertex_double2", 11, 0, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setVertexProperty<double>("vertex_double2", 11, 0, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setVertexProperty<int, 3>("vertex_int3", 11, { })), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeSource(11, 0), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeDestination(11, 0), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeSourceDestination(11, 0, 0), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeProperty<int>("edge_int", 11, static_cast<int>(0)), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setEdgeProperty<double, 2>("edge_double2", 11, 0, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW(graph.setEdgeProperty<double>("edge_double2", 11, 0, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW((graph.setEdgeProperty<float, 3>("edge_float3", 11, { })), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getVertexID(11), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getVertexProperty<float>("vertex_float", 11), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getVertexProperty<double, 2>("vertex_double2", 11, 0)), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getVertexProperty<double>("vertex_double2", 11, 0), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getVertexProperty<int, 3>("vertex_int3", 11)), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeSource(11), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeDestination(11), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeSourceDestination(11), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeProperty<int>("edge_int", 11), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getEdgeProperty<double, 2>("edge_double2", 11, 0)), exception::OutOfBoundsException);
    EXPECT_THROW(graph.getEdgeProperty<double>("edge_double2", 11, 0), exception::OutOfBoundsException);
    EXPECT_THROW((graph.getEdgeProperty<float, 3>("edge_float3", 11)), exception::OutOfBoundsException);
}
TEST(TestEnvironmentDirectedGraph, TestHostException) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with junk data
    model.newLayer().addHostFunction(HostException);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent1, MessageNone, MessageNone) {
    if (FLAMEGPU->getID() <= 20) {
        DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
        if (FLAMEGPU->getID() <= 10) {
            FLAMEGPU->setVariable<id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
            // vertex_int3, device full array access not available, so skipped
        }
        FLAMEGPU->setVariable<id_t>("edge_source", graph.getEdgeSource(FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<id_t>("edge_dest", graph.getEdgeDestination(FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
        // edge_float3, device full array access not available, so skipped
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent3, MessageNone, MessageNone) {
    if (FLAMEGPU->getID() <= 60) {
        DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
        if (FLAMEGPU->getID() <= 30) {
            FLAMEGPU->setVariable<id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
            // vertex_int3, device full array access not available, so skipped
        }
        FLAMEGPU->setVariable<id_t>("edge_source", graph.getEdgeSource(FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<id_t>("edge_dest", graph.getEdgeDestination(FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
        // edge_float3, device full array access not available, so skipped
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(HostDeviceCheckGraph) {
    DeviceAgentVector agent = FLAMEGPU->agent("agent").getPopulationData();
    for (unsigned int i = 0; i < 20; ++i) {
        if (i < 10) {
            EXPECT_EQ(agent[i].getVariable<id_t>("vertex_id"), static_cast<id_t>(i));
            EXPECT_EQ(agent[i].getVariable<float>("vertex_float"), static_cast<float>(i));
            std::array<double, 2> result = { static_cast<double>(i + 11), static_cast<double>(i + 21) };
            EXPECT_EQ((agent[i].getVariable<double, 2>("vertex_double2")), result);
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 2) % 10));
        } else {
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 6) % 10));
        }
        // Edges
        EXPECT_EQ(agent[i].getVariable<id_t>("edge_source"), static_cast<id_t>(i / 2));
        EXPECT_EQ(agent[i].getVariable<int>("edge_int"), static_cast<int>(i + 70));
        std::array<double, 2> result = { static_cast<double>(i + 61), static_cast<double>(i + 51) };
        EXPECT_EQ((agent[i].getVariable<double, 2>("edge_double2")), result);
    }
}
FLAMEGPU_HOST_FUNCTION(HostDeviceCheckGraph3) {
    DeviceAgentVector agent = FLAMEGPU->agent("agent").getPopulationData();
    for (unsigned int i = 0; i < 60; ++i) {
        if (i < 30) {
            EXPECT_EQ(agent[i].getVariable<id_t>("vertex_id"), static_cast<id_t>(i));
            EXPECT_EQ(agent[i].getVariable<float>("vertex_float"), static_cast<float>(i));
            std::array<double, 2> result = { static_cast<double>(i + 11), static_cast<double>(i + 21) };
            EXPECT_EQ((agent[i].getVariable<double, 2>("vertex_double2")), result);
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 2) % 30));
        } else {
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 18) % 30));
        }
        // Edges
        EXPECT_EQ(agent[i].getVariable<id_t>("edge_source"), static_cast<id_t>(i / 2));
        EXPECT_EQ(agent[i].getVariable<int>("edge_int"), static_cast<int>(i + 70));
        std::array<double, 2> result = { static_cast<double>(i + 61), static_cast<double>(i + 51) };
        EXPECT_EQ((agent[i].getVariable<double, 2>("edge_double2")), result);
    }
}
TEST(TestEnvironmentDirectedGraph, TestDeviceGetResetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");

    agent.newVariable<id_t>("vertex_id");
    agent.newVariable<float>("vertex_float");
    agent.newVariable<double, 2>("vertex_double2");
    // agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    agent.newVariable<id_t>("edge_source");
    agent.newVariable<id_t>("edge_dest");
    agent.newVariable<int>("edge_int");
    agent.newVariable<double, 2>("edge_double2");
    // agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped
    agent.newFunction("fn1", CopyGraphToAgent1);
    agent.newFunction("fn2", CopyGraphToAgent3);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);
    // Copy Data from Graph to Agent
    model.newLayer().addAgentFunction(CopyGraphToAgent1);
    // Check the agent data is correct persists
    model.newLayer().addHostFunction(HostDeviceCheckGraph);
    // Init graph with different known data
    model.newLayer().addHostFunction(InitGraph3);
    // Copy Data from Graph to Agent
    model.newLayer().addAgentFunction(CopyGraphToAgent3);
    // Check the agent data is correct persists
    model.newLayer().addHostFunction(HostDeviceCheckGraph3);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 60);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_EXIT_CONDITION(always_exit) {
    return flamegpu::EXIT;
}
TEST(TestEnvironmentDirectedGraph, SubModelSet_MasterModelHostGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");
    submodel.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    // Init graph with known data
    submodel.newLayer().addHostFunction(InitGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();

    // Init graph with known data
    model.newLayer().addSubModel(sub_desc);

    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, SubModelSet_MasterModelDeviceGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");

    AgentDescription agent = submodel.newAgent("agent");
    agent.newVariable<float>("foobar");  // Agent is not required in this model for this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    submodel.newLayer().addHostFunction(InitGraph);

    // Copy data to agent and check in host fn
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription master_agent = model.newAgent("agent");
    master_agent.newVariable<float>("foobar");  // Agent is only used to init a population
    master_agent.newVariable<id_t>("vertex_id");
    master_agent.newVariable<float>("vertex_float");
    master_agent.newVariable<double, 2>("vertex_double2");
    // master_agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    master_agent.newVariable<id_t>("edge_source");
    master_agent.newVariable<id_t>("edge_dest");
    master_agent.newVariable<int>("edge_int");
    master_agent.newVariable<double, 2>("edge_double2");
    // master_agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped

    master_agent.newFunction("fn1", CopyGraphToAgent1);

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();
    sub_desc.bindAgent("agent", "agent");

    // Init graph with known data
    model.newLayer().addSubModel(sub_desc);

    // Check the data persists
    model.newLayer().addAgentFunction(CopyGraphToAgent1);
    model.newLayer().addHostFunction(HostDeviceCheckGraph);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(master_agent, 20);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, MasterModelSet_SubModelHostGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");
    submodel.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    // Init graph with known data
    submodel.newLayer().addHostFunction(HostCheckGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);

    // Check the data persists
    model.newLayer().addSubModel(sub_desc);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, MasterModelSet_SubModelDeviceGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");

    AgentDescription agent = submodel.newAgent("agent");
    agent.newVariable<id_t>("vertex_id");
    agent.newVariable<float>("vertex_float");
    agent.newVariable<double, 2>("vertex_double2");
    // agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    agent.newVariable<id_t>("edge_source");
    agent.newVariable<id_t>("edge_dest");
    agent.newVariable<int>("edge_int");
    agent.newVariable<double, 2>("edge_double2");
    // agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped

    agent.newFunction("fn1", CopyGraphToAgent1);

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");


    // Copy data to agent and check in host fn
    submodel.newLayer().addAgentFunction(CopyGraphToAgent1);
    submodel.newLayer().addHostFunction(HostDeviceCheckGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription master_agent = model.newAgent("agent");
    master_agent.newVariable<float>("foobar");  // Agent is only used to init a population

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();
    sub_desc.bindAgent("agent", "agent");

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);

    // Check the data persists
    model.newLayer().addSubModel(sub_desc);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(master_agent, 20);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_HOST_FUNCTION(HostTestEdgesOut) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(5);
    graph.setEdgeCount(15);
    int k = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            graph.setEdgeProperty<id_t>("src_copy", k, j);
            graph.setEdgeSourceDestination(k++, j, i);
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(IterateEdgesOut, MessageNone, MessageNone) {
    id_t src = FLAMEGPU->getIndex();
    unsigned int ct = 0;
    bool src_all_correct = true;
    auto filter = FLAMEGPU->environment.getDirectedGraph("graph").outEdges(src);
    FLAMEGPU->setVariable<int>("count2", filter.size());
    for (auto &edge : filter) {
        src_all_correct &= edge.getProperty<id_t>("src_copy") == src;
        FLAMEGPU->setVariable<id_t, 5>("dests", ct, edge.getEdgeDestination());
        ++ct;
    }
    FLAMEGPU->setVariable<int>("count", ct);
    FLAMEGPU->setVariable<int>("src_all_correct", src_all_correct ? 1 : 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(HostTestEdgesIn) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(5);
    graph.setEdgeCount(15);
    int k = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
            graph.setEdgeProperty<id_t>("dest_copy", k, j);
            graph.setEdgeSourceDestination(k++, i, j);
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(IterateEdgesIn, MessageNone, MessageNone) {
    id_t dest = FLAMEGPU->getIndex();
    unsigned int ct = 0;
    bool dest_all_correct = true;
    auto filter = FLAMEGPU->environment.getDirectedGraph("graph").inEdges(dest);
    FLAMEGPU->setVariable<int>("count2", filter.size());
    for (auto& edge : filter) {
        dest_all_correct &= edge.getProperty<id_t>("dest_copy") == dest;
        FLAMEGPU->setVariable<id_t, 5>("srcs", ct, edge.getEdgeSource());
        ++ct;
    }
    FLAMEGPU->setVariable<int>("count", ct);
    FLAMEGPU->setVariable<int>("dest_all_correct", dest_all_correct ? 1 : 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(SaveGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Save
    graph.exportGraph("graph.json");
}
FLAMEGPU_HOST_FUNCTION(LoadCheckGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Load
    graph.importGraph("graph.json");
    // Check
    for (unsigned int i = 0; i < 60; ++i) {
        if (i < 30) {
            EXPECT_EQ(graph.getVertexID(i), static_cast<id_t>(i));
            EXPECT_EQ(graph.getVertexProperty<float>("vertex_float", i), static_cast<float>(i));
            std::array<double, 2> result = { static_cast<double>(i + 11), static_cast<double>(i + 21) };
            EXPECT_EQ((graph.getVertexProperty<double, 2>)("vertex_double2", i), result);
            EXPECT_EQ(graph.getEdgeDestination(i), static_cast<id_t>((i + 2) % 30));
        } else {
            EXPECT_EQ(graph.getEdgeDestination(i), static_cast<id_t>((i + 18) % 30));
        }
        // Edges
        EXPECT_EQ(graph.getEdgeSource(i), static_cast<id_t>(i / 2));
        EXPECT_EQ(graph.getEdgeProperty<int>("edge_int", i), static_cast<int>(i + 70));
        std::array<double, 2> result = { static_cast<double>(i + 61), static_cast<double>(i + 51) };
        EXPECT_EQ((graph.getEdgeProperty<double, 2>)("edge_double2", i), result);
    }
    // Cleanup
    std::filesystem::remove("graph.json");
}
TEST(TestEnvironmentDirectedGraph, TestJSONSaveLoad) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph3);
    // Export
    model.newLayer().addHostFunction(SaveGraph);
    // Reinit graph with different data
    model.newLayer().addHostFunction(InitGraph);
    // Import graph and check it matches first init
    model.newLayer().addHostFunction(LoadCheckGraph3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, TestEdgesOut) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newEdgeProperty<id_t>("src_copy");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t, 5>("dests");
    agent.newVariable<int>("count");
    agent.newVariable<int>("count2");
    agent.newVariable<int>("src_all_correct");
    agent.newFunction("iterate_edges", IterateEdgesOut);

    // Init graph with known data
    model.newLayer().addHostFunction(HostTestEdgesOut);
    model.newLayer().addAgentFunction(IterateEdgesOut);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 5);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    int k = 0;
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<int>("src_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("count"), 5 - k);
        EXPECT_EQ(agt.getVariable<int>("count2"), 5 - k);
        for (int i = 0; i < 5 - k; ++i) {
            EXPECT_EQ(agt.getVariable<id_t>("dests", i), static_cast<id_t>(k + i));
        }
        ++k;
    }
}
TEST(TestEnvironmentDirectedGraph, TestEdgesIn) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newEdgeProperty<id_t>("dest_copy");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t, 5>("srcs");
    agent.newVariable<int>("count");
    agent.newVariable<int>("count2");
    agent.newVariable<int>("dest_all_correct");
    agent.newFunction("iterate_edges", IterateEdgesIn);

    // Init graph with known data
    model.newLayer().addHostFunction(HostTestEdgesIn);
    model.newLayer().addAgentFunction(IterateEdgesIn);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 5);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    int k = 0;
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<int>("dest_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("count"), 5 - k);
        EXPECT_EQ(agt.getVariable<int>("count2"), 5 - k);
        for (int i = 0; i < 5 - k; ++i) {
            EXPECT_EQ(agt.getVariable<id_t>("srcs", i), static_cast<id_t>(k + i));
        }
        ++k;
    }
}
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS1, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<float>("vertex_float", 10);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS2, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<double, 2>("vertex_double2", 10, 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS3, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexID(10);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS1, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<int>("edge_int", 20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS2, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<double, 2>("edge_double2", 20, 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS3, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeSource(20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS4, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeSource(20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceOutEdges_SEATBELTS, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").outEdges(10);
    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS1);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS1);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS2) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS2);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS2);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS3) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS3);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS1);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS1);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS2) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS2);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS2);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS3) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS3);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS4) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS4);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS4);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceOutEdgesLeaving_SEATBELTS) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceOutEdges_SEATBELTS);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceOutEdges_SEATBELTS);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph_RTC, Test_DeviceGetVertex_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    const char *DeviceGetVertex_SEATBELTS1_RTC = R"###(
    FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS1, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<float>("vertex_float", 10);
        return flamegpu::ALIVE;
    }
    )###";
    auto t = agent.newRTCFunction("fn1", DeviceGetVertex_SEATBELTS1_RTC);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(t);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph_RTC, Test_DeviceGetEdge_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    const char* DeviceGetEdge_SEATBELTS1_RTC = R"###(
    FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS1, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<int>("edge_int", 20);
        return flamegpu::ALIVE;
    }
    )###";
    auto t = agent.newRTCFunction("fn1", DeviceGetEdge_SEATBELTS1_RTC);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(t);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
#else
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS2) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS3) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS2) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS3) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS4) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceOutEdgesLeaving_SEATBELTS) { }
TEST(TestEnvironmentDirectedGraph_RTC, DISABLED_Test_DeviceGetVertex_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph_RTC, DISABLED_Test_DeviceGetEdge_SEATBELTS1) { }
#endif

}  // namespace test_environment_directed_graph
}  // namespace flamegpu
