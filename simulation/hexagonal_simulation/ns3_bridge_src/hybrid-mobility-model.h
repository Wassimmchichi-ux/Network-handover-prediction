#ifndef HYBRID_MOBILITY_MODEL_H
#define HYBRID_MOBILITY_MODEL_H

#include "ns3/box.h"
#include "ns3/mobility-model.h"
#include "ns3/nstime.h"
#include "ns3/vector.h"

namespace ns3 {

class HybridMobilityModel : public MobilityModel
{
public:
  static TypeId GetTypeId (void);

  HybridMobilityModel ();
  ~HybridMobilityModel () override;

  void SetBounds (const Box& bounds);
  void SetWaypoint (const Vector& waypoint);
  void ClearWaypoint ();
  void SetVelocity (const Vector& velocity);
  void UpdateVelocity (double vx, double vy, double vz = 0.0);
  void Advance (Time dt);

protected:
  Vector DoGetPosition (void) const override;
  void DoSetPosition (const Vector& position) override;
  Vector DoGetVelocity (void) const override;

private:
  void ClampOrBounce ();

  Vector m_position;
  Vector m_velocity;
  Vector m_waypoint;
  bool m_hasWaypoint;
  bool m_hasBounds;
  Box m_bounds;
};

} // namespace ns3

#endif
