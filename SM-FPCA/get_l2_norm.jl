function getL2Norm(f, R0)


norm = sqrt(f' * R0 * f)

return norm
end