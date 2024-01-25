import argparse

def main():
	parser = argparse.ArgumentParser(description='Calculate order or cardinality of an elliptic curve.')
	parser.add_argument('p', type=int, help='Prime number p')
	parser.add_argument('a', type=int, help='Coefficient a of the elliptic curve')
	parser.add_argument('b', type=int, help='Coefficient b of the elliptic curve')
	parser.add_argument('-x', type=int, default=None, help='x-coordinate of the point')
	parser.add_argument('-y', type=int, default=None, help='y-coordinate of the point')
	parser.add_argument('-o', '--order', action='store_true', help='Calculate the order of a point')
	parser.add_argument('-c', '--cardinality', action='store_true', help='Calculate the cardinality of the curve')
	parser.add_argument('-A', '--all', action='store_true', help='Calculate both order and cardinality')

	args = parser.parse_args()
	E = EllipticCurve(GF(args.p), [args.a, args.b])

	if args.all:
		if args.x is None or args.y is None:
			raise ValueError("x and y coordinates must be provided for order calculation")
		
		P = E(args.x, args.y)
		print(P.order(), E.cardinality())

	if args.order:
		if args.x is None or args.y is None:
			raise ValueError("x and y coordinates must be provided for order calculation")
		
		P = E(args.x, args.y)
		print(P.order())

	if args.cardinality:
		print(E.cardinality())
		


if __name__ == "__main__":
	main()


	