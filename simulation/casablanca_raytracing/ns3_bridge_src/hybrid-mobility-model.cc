#include "hybrid-mobility-model.h"

#include "ns3/log.h"

#include <algorithm>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HybridMobilityModel");
NS_OBJECT_ENSURE_REGISTERED (HybridMobilityModel);

TypeId
HybridMobilityModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::HybridMobilityModel")
    .SetParent<MobilityModel> ()
    .SetGroupName ("Mobility")
    .AddConstructor<HybridMobilityModel> ();
  return tid;
}

HybridMobilityModel::HybridMobilityModel ()
  : m_position (0.0, 0.0, 0.0),
    m_velocity (0.0, 0.0, 0.0),
    m_waypoint (0.0, 0.0, 0.0),
    m_hasWaypoint (false),
    m_hasBounds (false),
    m_bounds ()
{
}

HybridMobilityModel::~HybridMobilityModel () = default;

void
HybridMobilityModel::SetBounds (const Box& bounds)
{
  m_bounds = bounds;
  m_hasBounds = true;
  ClampOrBounce ();
}

void
HybridMobilityModel::SetWaypoint (const Vector& waypoint)
{
  m_waypoint = waypoint;
  m_hasWaypoint = true;
}

void
HybridMobilityModel::ClearWaypoint ()
{
  m_hasWaypoint = false;
}

void
HybridMobilityModel::SetVelocity (const Vector& velocity)
{
  m_velocity = velocity;
  NotifyCourseChange ();
}

void
HybridMobilityModel::UpdateVelocity (double vx, double vy, double vz)
{
  SetVelocity (Vector (vx, vy, vz));
}

void
HybridMobilityModel::Advance (Time dt)
{
  const double seconds = dt.GetSeconds ();
  if (seconds <= 0.0)
    {
      return;
    }

  if (m_hasWaypoint)
    {
      const double dx = m_waypoint.x - m_position.x;
      const double dy = m_waypoint.y - m_position.y;
      const double dz = m_waypoint.z - m_position.z;
      const double distance = std::sqrt ((dx * dx) + (dy * dy) + (dz * dz));
      const double speed = std::sqrt ((m_velocity.x * m_velocity.x) +
                                      (m_velocity.y * m_velocity.y) +
                                      (m_velocity.z * m_velocity.z));
      if (distance > 0.0 && speed > 0.0)
        {
          const double maxStep = speed * seconds;
          if (distance <= maxStep)
            {
              m_position = m_waypoint;
              m_velocity = Vector (0.0, 0.0, 0.0);
              m_hasWaypoint = false;
            }
          else
            {
              const double scale = maxStep / distance;
              m_position.x += dx * scale;
              m_position.y += dy * scale;
              m_position.z += dz * scale;
            }
        }
    }
  else
    {
      m_position.x += m_velocity.x * seconds;
      m_position.y += m_velocity.y * seconds;
      m_position.z += m_velocity.z * seconds;
    }

  ClampOrBounce ();
  NotifyCourseChange ();
}

Vector
HybridMobilityModel::DoGetPosition (void) const
{
  return m_position;
}

void
HybridMobilityModel::DoSetPosition (const Vector& position)
{
  m_position = position;
  ClampOrBounce ();
  NotifyCourseChange ();
}

Vector
HybridMobilityModel::DoGetVelocity (void) const
{
  return m_velocity;
}

void
HybridMobilityModel::ClampOrBounce ()
{
  if (!m_hasBounds)
    {
      return;
    }

  if (m_position.x < m_bounds.xMin || m_position.x > m_bounds.xMax)
    {
      m_velocity.x *= -1.0;
      m_position.x = std::min (std::max (m_position.x, m_bounds.xMin), m_bounds.xMax);
    }
  if (m_position.y < m_bounds.yMin || m_position.y > m_bounds.yMax)
    {
      m_velocity.y *= -1.0;
      m_position.y = std::min (std::max (m_position.y, m_bounds.yMin), m_bounds.yMax);
    }
  if (m_position.z < m_bounds.zMin || m_position.z > m_bounds.zMax)
    {
      m_velocity.z *= -1.0;
      m_position.z = std::min (std::max (m_position.z, m_bounds.zMin), m_bounds.zMax);
    }
}

} // namespace ns3
