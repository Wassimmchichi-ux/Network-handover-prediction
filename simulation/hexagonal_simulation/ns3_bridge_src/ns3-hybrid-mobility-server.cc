#include "hybrid-mobility-model.h"
#include "minizmq.h"

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ns3;

namespace {

static const std::uint32_t STATE_MAGIC = 0x31425948;   // HYB1
static const std::uint32_t CONTROL_MAGIC = 0x32425948; // HYB2

struct CellRow
{
  std::uint32_t cellId;
  double x;
  double y;
  double altitude;
};

struct UeRow
{
  std::string ueId;
  int scenarioId;
  std::string mobilityType;
  double x;
  double y;
  double z;
  double vx;
  double vy;
  double vz;
};

struct DroneRow
{
  std::uint32_t cellId;
  double x;
  double y;
  double altitude;
  double speed;
};

template <typename T>
void
AppendPod (std::vector<std::uint8_t>& buffer, const T& value)
{
  const auto* bytes = reinterpret_cast<const std::uint8_t*> (&value);
  buffer.insert (buffer.end (), bytes, bytes + sizeof (T));
}

template <typename T>
T
ReadPod (const std::vector<std::uint8_t>& buffer, std::size_t& offset)
{
  if (offset + sizeof (T) > buffer.size ())
    {
      throw std::runtime_error ("Truncated control packet");
    }
  T value;
  std::memcpy (&value, buffer.data () + offset, sizeof (T));
  offset += sizeof (T);
  return value;
}

std::vector<std::string>
SplitCsv (const std::string& line)
{
  std::vector<std::string> out;
  std::stringstream ss (line);
  std::string token;
  while (std::getline (ss, token, ','))
    {
      out.push_back (token);
    }
  return out;
}

std::vector<CellRow>
LoadCells (const std::string& path)
{
  std::ifstream handle (path);
  if (!handle.is_open ())
    {
      throw std::runtime_error ("Cannot open active cells CSV: " + path);
    }
  std::string line;
  std::getline (handle, line);
  std::vector<CellRow> rows;
  while (std::getline (handle, line))
    {
      if (line.empty ())
        {
          continue;
        }
      auto cols = SplitCsv (line);
      if (cols.size () < 4)
        {
          continue;
        }
      rows.push_back ({static_cast<std::uint32_t> (std::stoul (cols[0])),
                       std::stod (cols[1]),
                       std::stod (cols[2]),
                       std::stod (cols[3])});
    }
  return rows;
}

std::vector<DroneRow>
LoadDrones (const std::string& path)
{
  std::ifstream handle (path);
  if (!handle.is_open ())
    {
      throw std::runtime_error ("Cannot open active drones CSV: " + path);
    }
  std::string line;
  std::getline (handle, line);
  std::vector<DroneRow> rows;
  while (std::getline (handle, line))
    {
      if (line.empty ())
        {
          continue;
        }
      auto cols = SplitCsv (line);
      if (cols.size () < 7)
        {
          continue;
        }
      rows.push_back ({static_cast<std::uint32_t> (std::stoul (cols[0])),
                       std::stod (cols[1]),
                       std::stod (cols[2]),
                       std::stod (cols[3]),
                       std::stod (cols[6])});
    }
  return rows;
}

std::vector<UeRow>
LoadUes (const std::string& path)
{
  std::ifstream handle (path);
  if (!handle.is_open ())
    {
      throw std::runtime_error ("Cannot open active UEs CSV: " + path);
    }
  std::string line;
  std::getline (handle, line);
  std::vector<UeRow> rows;
  while (std::getline (handle, line))
    {
      if (line.empty ())
        {
          continue;
        }
      auto cols = SplitCsv (line);
      if (cols.size () < 9)
        {
          continue;
        }
      rows.push_back ({cols[0],
                       std::stoi (cols[1]),
                       cols[2],
                       std::stod (cols[3]),
                       std::stod (cols[4]),
                       std::stod (cols[5]),
                       std::stod (cols[6]),
                       std::stod (cols[7]),
                       std::stod (cols[8])});
    }
  return rows;
}

void
ConfigureSocket (void* socket, int option, int value)
{
  if (zmq_setsockopt (socket, option, &value, sizeof (value)) != 0)
    {
      throw std::runtime_error (std::string ("zmq_setsockopt failed: ") + zmq_strerror (zmq_errno ()));
    }
}

} // namespace

int
main (int argc, char* argv[])
{
  std::string activeCells = "active_ground_cells.csv";
  std::string activeDrones = "active_drones.csv";
  std::string activeUes = "active_ues.csv";
  std::string endpoint = "tcp://127.0.0.1:5557";
  double simTime = 600.0;
  std::uint32_t dtMs = 10;

  CommandLine cmd;
  cmd.AddValue ("activeCells", "CSV with selected ground cells", activeCells);
  cmd.AddValue ("activeDrones", "CSV with initial drone states", activeDrones);
  cmd.AddValue ("activeUes", "CSV with initial UE states", activeUes);
  cmd.AddValue ("endpoint", "ZeroMQ endpoint", endpoint);
  cmd.AddValue ("simTime", "Simulation time in seconds", simTime);
  cmd.AddValue ("dtMs", "Mobility step in milliseconds", dtMs);
  cmd.Parse (argc, argv);

  auto cellRows = LoadCells (activeCells);
  auto droneRows = LoadDrones (activeDrones);
  auto ueRows = LoadUes (activeUes);

  if (cellRows.empty () || ueRows.empty ())
    {
      std::cerr << "Active cell or UE list is empty.\n";
      return 1;
    }

  double xmin = cellRows.front ().x;
  double xmax = cellRows.front ().x;
  double ymin = cellRows.front ().y;
  double ymax = cellRows.front ().y;
  for (const auto& row : cellRows)
    {
      xmin = std::min (xmin, row.x);
      xmax = std::max (xmax, row.x);
      ymin = std::min (ymin, row.y);
      ymax = std::max (ymax, row.y);
    }
  const Box bounds (xmin - 250.0, xmax + 250.0, ymin - 250.0, ymax + 250.0, 0.0, 500.0);

  std::vector<Ptr<HybridMobilityModel>> ueModels;
  ueModels.reserve (ueRows.size ());
  for (const auto& row : ueRows)
    {
      auto model = CreateObject<HybridMobilityModel> ();
      model->SetBounds (bounds);
      model->SetPosition (Vector (row.x, row.y, row.z));
      model->SetVelocity (Vector (row.vx, row.vy, row.vz));
      ueModels.push_back (model);
    }

  std::vector<Ptr<HybridMobilityModel>> droneModels;
  droneModels.reserve (droneRows.size ());
  for (const auto& row : droneRows)
    {
      auto model = CreateObject<HybridMobilityModel> ();
      model->SetBounds (bounds);
      model->SetPosition (Vector (row.x, row.y, row.altitude));
      model->SetVelocity (Vector (row.speed, 0.0, 0.0));
      model->SetWaypoint (Vector (row.x, row.y, row.altitude));
      droneModels.push_back (model);
    }

  void* context = zmq_ctx_new ();
  if (!context)
    {
      std::cerr << "zmq_ctx_new failed: " << zmq_strerror (zmq_errno ()) << "\n";
      return 1;
    }
  void* socket = zmq_socket (context, ZMQ_REQ);
  if (!socket)
    {
      std::cerr << "zmq_socket failed: " << zmq_strerror (zmq_errno ()) << "\n";
      zmq_ctx_term (context);
      return 1;
    }

  try
    {
      ConfigureSocket (socket, ZMQ_LINGER, 0);
      ConfigureSocket (socket, ZMQ_RCVTIMEO, 60000);
      ConfigureSocket (socket, ZMQ_SNDTIMEO, 60000);
      if (zmq_connect (socket, endpoint.c_str ()) != 0)
        {
          throw std::runtime_error (std::string ("zmq_connect failed: ") + zmq_strerror (zmq_errno ()));
        }

      const std::uint32_t totalSteps = static_cast<std::uint32_t> (std::llround (simTime * 1000.0 / dtMs));
      const Time dt = MilliSeconds (dtMs);
      for (std::uint32_t step = 0; step <= totalSteps; ++step)
        {
          const std::uint32_t timeMs = step * dtMs;
          std::vector<std::uint8_t> payload;
          payload.reserve (20 + (ueModels.size () + droneModels.size ()) * 24);
          AppendPod (payload, STATE_MAGIC);
          AppendPod (payload, step);
          AppendPod (payload, timeMs);
          AppendPod (payload, static_cast<std::uint32_t> (ueModels.size ()));
          AppendPod (payload, static_cast<std::uint32_t> (droneModels.size ()));

          for (const auto& model : ueModels)
            {
              Vector pos = model->GetPosition ();
              Vector vel = model->GetVelocity ();
              AppendPod (payload, static_cast<float> (pos.x));
              AppendPod (payload, static_cast<float> (pos.y));
              AppendPod (payload, static_cast<float> (pos.z));
              AppendPod (payload, static_cast<float> (vel.x));
              AppendPod (payload, static_cast<float> (vel.y));
              AppendPod (payload, static_cast<float> (vel.z));
            }
          for (const auto& model : droneModels)
            {
              Vector pos = model->GetPosition ();
              Vector vel = model->GetVelocity ();
              AppendPod (payload, static_cast<float> (pos.x));
              AppendPod (payload, static_cast<float> (pos.y));
              AppendPod (payload, static_cast<float> (pos.z));
              AppendPod (payload, static_cast<float> (vel.x));
              AppendPod (payload, static_cast<float> (vel.y));
              AppendPod (payload, static_cast<float> (vel.z));
            }

          if (zmq_send (socket, payload.data (), payload.size (), 0) < 0)
            {
              throw std::runtime_error (std::string ("zmq_send failed: ") + zmq_strerror (zmq_errno ()));
            }

          std::vector<std::uint8_t> recvBuffer (65536);
          const int recvBytes = zmq_recv (socket, recvBuffer.data (), recvBuffer.size (), 0);
          if (recvBytes < 0)
            {
              throw std::runtime_error (std::string ("zmq_recv failed: ") + zmq_strerror (zmq_errno ()));
            }
          recvBuffer.resize (static_cast<std::size_t> (recvBytes));

          std::size_t offset = 0;
          const std::uint32_t magic = ReadPod<std::uint32_t> (recvBuffer, offset);
          if (magic != CONTROL_MAGIC)
            {
              throw std::runtime_error ("Unexpected control packet magic");
            }
          const std::uint32_t controlStep = ReadPod<std::uint32_t> (recvBuffer, offset);
          const std::uint32_t stopFlag = ReadPod<std::uint32_t> (recvBuffer, offset);
          const std::uint32_t ueUpdates = ReadPod<std::uint32_t> (recvBuffer, offset);
          const std::uint32_t droneWaypoints = ReadPod<std::uint32_t> (recvBuffer, offset);
          if (controlStep != step)
            {
              throw std::runtime_error ("Control step mismatch");
            }

          for (std::uint32_t i = 0; i < ueUpdates; ++i)
            {
              const std::uint32_t index = ReadPod<std::uint32_t> (recvBuffer, offset);
              const float vx = ReadPod<float> (recvBuffer, offset);
              const float vy = ReadPod<float> (recvBuffer, offset);
              const float vz = ReadPod<float> (recvBuffer, offset);
              if (index < ueModels.size ())
                {
                  ueModels[index]->UpdateVelocity (vx, vy, vz);
                }
            }

          for (std::uint32_t i = 0; i < droneWaypoints; ++i)
            {
              const std::uint32_t index = ReadPod<std::uint32_t> (recvBuffer, offset);
              const float x = ReadPod<float> (recvBuffer, offset);
              const float y = ReadPod<float> (recvBuffer, offset);
              const float z = ReadPod<float> (recvBuffer, offset);
              const float speed = ReadPod<float> (recvBuffer, offset);
              if (index < droneModels.size ())
                {
                  droneModels[index]->SetWaypoint (Vector (x, y, z));
                  droneModels[index]->SetVelocity (Vector (speed, 0.0, 0.0));
                }
            }

          if (stopFlag || step == totalSteps)
            {
              break;
            }

          for (auto& model : ueModels)
            {
              model->Advance (dt);
            }
          for (auto& model : droneModels)
            {
              model->Advance (dt);
            }
        }
    }
  catch (const std::exception& ex)
    {
      std::cerr << "ns-3 bridge error: " << ex.what () << "\n";
      zmq_close (socket);
      zmq_ctx_term (context);
      return 1;
    }

  zmq_close (socket);
  zmq_ctx_term (context);
  return 0;
}
