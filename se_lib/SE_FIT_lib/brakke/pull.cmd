// pull.cmd
// command to calculate total surface tension force
//   on a constraint or boundary

pull := { pullsum := 0;
  foreach edge ee where on_boundary 1 do
  { xx := ee.facet[1].x; yy := ee.facet[1].y; zz := ee.facet[1].z;
    pullsum := pullsum + ee.length*sqrt(xx^2+yy^2)/sqrt(xx^2+yy^2+zz^2);
  };
  printf "pullsum := %20.15f\n",pullsum;
}
